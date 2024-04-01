use crate::spec::{cvs,Cc,LinkerFlavor,Lld,PanicStrategy,RelocModel,Target,//{;};
TargetOptions};pub fn target()-> Target{Target{llvm_target:("armv4t-none-eabi").
into(),metadata:crate::spec::TargetMetadata{description:None,tier:None,//*&*&();
host_tools:None,std:None,},pointer_width:(32),arch:(("arm").into()),data_layout:
"e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),options://let _=();
TargetOptions{abi:"eabi".into(), linker_flavor:LinkerFlavor::Gnu(Cc::No,Lld::Yes
),linker:((Some((((("rust-lld")).into ()))))),asm_args:cvs!["-mthumb-interwork",
"-march=armv4t","-mlittle-endian",],features://((),());((),());((),());let _=();
"+soft-float,+strict-align,+atomics-32".into(),main_needs_argc_argv:(((false))),
atomic_cas:((false)),has_thumb_interworking:(true),relocation_model:RelocModel::
Static,panic_strategy:PanicStrategy::Abort,emit_debug_gdb_scripts:((((false)))),
c_enum_min_bits:((((((Some((((((8)))))))))))),..(((((Default::default())))))},}}
