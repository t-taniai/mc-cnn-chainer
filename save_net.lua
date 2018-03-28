-- This is a script that was used for saving MC-CNN network weights
-- as binary files in /mccnn

#! /usr/bin/env luajit

require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'

cmd = torch.CmdLine()
cmd:option('-net_fname', '')
cmd:option('-out_fname', '')
cmd:option('-arch', '')
opt = cmd:parse(arg)


include('SpatialConvolution1_fw.lua')
include('Normalize2.lua')
include('StereoJoin.lua')

function writeTensor(t, fname)
fp = torch.DiskFile(fname, 'w')
fp:binary()
fp:writeInt(t:nDimension())
for i = 1, t:nDimension() do
fp:writeInt(t:size(i))
end
fp:writeFloat(t:float():storage())
fp:close()
end

function printSize(t)
for i = 1, t:nDimension() do
io.write(t:size(i))
io.write(' ')
end
print('')
end


   if opt.arch == 'slow' then
      net = torch.load(opt.net_fname, 'ascii')
      net_te = net[1]
      net_te2 = net[2]
   elseif opt.arch == 'fast' then
      net_te = torch.load(opt.net_fname, 'ascii')[1]
   end

   c = 1
   for i = 1,#net_te.modules do
      local module = net_te.modules[i]
      if torch.typename(module) == 'cudnn.SpatialConvolution' then
         W = module.weight
         B = module.bias
         print(('--- layer %d ---'):format(c))
         printSize(W)
         printSize(B)
         --torch.DiskFile(('%s_1_%dW.bin'):format(opt.out_fname, c), 'w'):binary():writeFloat(W:float():storage())
         --torch.DiskFile(('%s_1_%dB.bin'):format(opt.out_fname, c), 'w'):binary():writeFloat(B:float():storage())
         writeTensor(W, ('%s_1_%dW.bin'):format(opt.out_fname, c))
         writeTensor(B, ('%s_1_%dB.bin'):format(opt.out_fname, c))
         c = c + 1
      end
   end

   if opt.arch == 'slow' then
   c = 1
   print('')
   for i = 1,#net_te2.modules do
      local module = net_te2.modules[i]
      if torch.typename(module) == 'nn.SpatialConvolution1_fw' then
      --if torch.typename(module) == 'cudnn.SpatialConvolution' then
         W = module.weight
         B = module.bias
         W = W:reshape(torch.LongStorage({W:size(1), W:size(2), 1, 1}))
         B = B:reshape(torch.LongStorage({B:size(2)}))
         print(('--- layer %d ---'):format(c))
         printSize(W)
         printSize(B)
         --torch.DiskFile(('%s_2_%dW.bin'):format(opt.out_fname, c), 'w'):binary():writeFloat(W:float():storage())
         --torch.DiskFile(('%s_2_%dB.bin'):format(opt.out_fname, c), 'w'):binary():writeFloat(B:float():storage())
         writeTensor(W, ('%s_2_%dW.bin'):format(opt.out_fname, c))
         writeTensor(B, ('%s_2_%dB.bin'):format(opt.out_fname, c))
         c = c + 1
      end
   end
   end
