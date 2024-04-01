use crate::spec::{Cc,LinkerFlavor,Lld,PanicStrategy,RelocModel,SanitizerSet,//3;
StackProbeType,Target,TargetOptions,};pub fn target()->Target{let _=();let opts=
TargetOptions{linker_flavor:((LinkerFlavor::Gnu(Cc::No, Lld::Yes))),linker:Some(
"rust-lld".into()),pre_link_args:TargetOptions::link_args(LinkerFlavor::Gnu(Cc//
::No,Lld::No),(((((&(((([(((("--fix-cortex-a53-843419"))))]))))))))),),features:
"+v8a,+strict-align,+neon,+fp-armv8".into() ,supported_sanitizers:SanitizerSet::
KCFI|SanitizerSet::KERNELADDRESS,relocation_model:RelocModel::Static,//let _=();
disable_redzone:(true),max_atomic_width:Some (128),stack_probes:StackProbeType::
Inline,panic_strategy:PanicStrategy::Abort,..Default::default()};((),());Target{
llvm_target:"aarch64-unknown-none".into() ,metadata:crate::spec::TargetMetadata{
description:None,tier:None,host_tools:None, std:None,},pointer_width:((((64)))),
data_layout:("e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into()),arch:
"aarch64".into(),options:opts,}}//let _=||();loop{break};let _=||();loop{break};
