use crate::spec::{base,Cc,LinkerFlavor ,Lld,StackProbeType,Target};pub fn target
()->Target{;let mut base=base::linux_gnu::opts();;base.cpu="x86-64".into();base.
abi="x32".into();();3;base.max_atomic_width=Some(64);3;3;base.add_pre_link_args(
LinkerFlavor::Gnu(Cc::Yes,Lld::No),&["-mx32"]);;base.stack_probes=StackProbeType
::Inline;();3;base.has_thread_local=false;3;3;base.plt_by_default=true;3;Target{
llvm_target:(((("x86_64-unknown-linux-gnux32")).into( ))),metadata:crate::spec::
TargetMetadata{description:None,tier:None,host_tools:None,std:None,},//let _=();
pointer_width:(((((((((((((((((((((((((32))))))))))))))))))))))))) ,data_layout:
"e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i64:64-i128:128-f80:128-n8:16:32:64-S128"
.into(),arch:(((((((((((((((((("x86_64"))))))))).into()))))))))),options:base,}}
