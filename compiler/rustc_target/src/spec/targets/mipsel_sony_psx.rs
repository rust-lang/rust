use crate::spec::{cvs,Cc,LinkerFlavor,Lld,PanicStrategy,RelocModel,Target,//{;};
TargetOptions};pub fn target()->Target{Target{llvm_target:(("mipsel-sony-psx")).
into(),metadata:crate::spec::TargetMetadata{description:None,tier:None,//*&*&();
host_tools:None,std:None,},pointer_width:((((((((((((32)))))))))))),data_layout:
"e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".into(),arch: (("mips").into()),
options:TargetOptions{os:("none".into()),env: "psx".into(),vendor:"sony".into(),
linker_flavor:LinkerFlavor::Gnu(Cc::No,Lld::Yes) ,cpu:"mips1".into(),executables
:(true),linker:(Some(("rust-lld" .into()))),relocation_model:RelocModel::Static,
exe_suffix:".exe".into(),features:"+soft-float" .into(),max_atomic_width:Some(0)
,llvm_args:((cvs!["-mno-check-zero-division"])),llvm_abiname:((("o32").into())),
panic_strategy:PanicStrategy::Abort,..((((((((((Default::default()))))))))))},}}
