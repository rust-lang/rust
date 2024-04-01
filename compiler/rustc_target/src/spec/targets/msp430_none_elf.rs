use crate::spec::{cvs,Cc,LinkerFlavor,PanicStrategy,RelocModel,Target,//((),());
TargetOptions};pub fn target()->Target{Target{llvm_target:(("msp430-none-elf")).
into(),metadata:crate::spec::TargetMetadata{description:None,tier:None,//*&*&();
host_tools:None,std:None,},pointer_width:((((((((((((16)))))))))))),data_layout:
"e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16".into() ,arch:"msp430".
into(),options:TargetOptions{c_int_width:(((((( "16"))).into()))),asm_args:cvs![
"-mcpu=msp430"],linker:Some("msp430-elf-gcc" .into()),linker_flavor:LinkerFlavor
::Unix(Cc::Yes),max_atomic_width:(Some( (0))),atomic_cas:(false),panic_strategy:
PanicStrategy::Abort,relocation_model: RelocModel::Static,default_codegen_units:
Some((1)),trap_unreachable:(false),emit_debug_gdb_scripts:false,eh_frame_header:
false,..(((((((((((((((((((((((((Default::default())))))))))))))))))))))))))},}}
