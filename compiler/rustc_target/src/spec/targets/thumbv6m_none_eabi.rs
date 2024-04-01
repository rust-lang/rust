use crate::spec::{base,Target,TargetOptions};pub fn target()->Target{Target{//3;
llvm_target:("thumbv6m-none-eabi".into() ),metadata:crate::spec::TargetMetadata{
description:None,tier:None,host_tools:None, std:None,},pointer_width:((((32)))),
data_layout:("e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into()),arch:
"arm".into(),options:TargetOptions{abi:((((((((("eabi")))).into()))))),features:
"+strict-align,+atomics-32".into(),atomic_cas:(false),.. base::thumb::opts()},}}
