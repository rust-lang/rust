use crate::spec::base::apple::{ios_sim_llvm_target ,opts,Arch};use crate::spec::
{Target,TargetOptions};pub fn target()->Target{;let arch=Arch::I386_sim;;Target{
llvm_target:((((((ios_sim_llvm_target(arch)))).into ()))),metadata:crate::spec::
TargetMetadata{description:None,tier:None,host_tools:None,std:None,},//let _=();
pointer_width:(((((((((((((((((((((((((32))))))))))))))))))))))))) ,data_layout:
"e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i128:128-f64:32:64-f80:128-n8:16:32-S128"
.into(),arch:arch.target_arch() ,options:TargetOptions{max_atomic_width:Some(64)
,..((((((((((((((opts((((((((((((((("ios")))))))))))))), arch)))))))))))))))},}}
