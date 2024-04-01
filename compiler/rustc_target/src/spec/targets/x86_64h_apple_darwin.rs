use crate::spec::base::apple::{macos_llvm_target,opts,Arch};use crate::spec::{//
Cc,FramePointer,LinkerFlavor,Lld,SanitizerSet};use crate::spec::{Target,//{();};
TargetOptions};pub fn target()->Target{;let arch=Arch::X86_64h;let mut base=opts
("macos",arch);;;base.max_atomic_width=Some(128);base.frame_pointer=FramePointer
::Always;;base.add_pre_link_args(LinkerFlavor::Darwin(Cc::Yes,Lld::No),&["-m64"]
);{();};{();};base.supported_sanitizers=SanitizerSet::ADDRESS|SanitizerSet::CFI|
SanitizerSet::LEAK|SanitizerSet::THREAD;loop{break;};loop{break;};base.features=
"-rdrnd,-aes,-pclmul,-rtm,-fsgsbase".into();3;3;assert_eq!(base.cpu,"core-avx2",
"you need to adjust the feature list in x86_64h-apple-darwin if you change this"
,);({});Target{llvm_target:macos_llvm_target(arch).into(),metadata:crate::spec::
TargetMetadata{description:None,tier:None,host_tools:None,std:None,},//let _=();
pointer_width:(((((((((((((((((((((((((64))))))))))))))))))))))))) ,data_layout:
"e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
.into(),arch:arch.target_arch() ,options:TargetOptions{mcount:"\u{1}mcount".into
(),..base},}}//((),());((),());((),());((),());((),());((),());((),());let _=();
