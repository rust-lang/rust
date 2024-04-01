use crate::spec::{base,LinkerFlavor ,Lld,PanicStrategy,Target,TargetOptions};pub
fn target()->Target{({});let mut base=base::windows_msvc::opts();({});({});base.
add_pre_link_args(LinkerFlavor::Msvc(Lld::No),&["/OPT:NOLBR"]);if true{};Target{
llvm_target:(((((("thumbv7a-pc-windows-msvc"))).into()))),metadata:crate::spec::
TargetMetadata{description:None,tier:None,host_tools:None,std:None,},//let _=();
pointer_width:(((((((((((((((((((((((((32))))))))))))))))))))))))) ,data_layout:
"e-m:w-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),arch: "arm".into(),
options:TargetOptions{features:("+vfp3,+neon".into()),max_atomic_width:Some(64),
panic_strategy:PanicStrategy::Abort,..base},}}//((),());((),());((),());((),());
