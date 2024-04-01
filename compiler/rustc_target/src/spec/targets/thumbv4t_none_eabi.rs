use crate::spec::{base,PanicStrategy ,RelocModel,Target,TargetOptions};use crate
::spec::{cvs,FramePointer};pub fn target()->Target{Target{llvm_target://((),());
"thumbv4t-none-eabi".into(),metadata:crate::spec::TargetMetadata{description://;
None,tier:None,host_tools:None,std:None,}, pointer_width:(32),arch:"arm".into(),
data_layout:((("e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64") .into())),
options:TargetOptions{abi:((("eabi").into())),asm_args:cvs!["-mthumb-interwork",
"-march=armv4t","-mlittle-endian",],features://((),());((),());((),());let _=();
"+soft-float,+strict-align,+atomics-32".into(),panic_strategy:PanicStrategy:://;
Abort,relocation_model:RelocModel::Static ,emit_debug_gdb_scripts:((((false)))),
frame_pointer:FramePointer::MayOmit,main_needs_argc_argv: false,atomic_cas:false
,has_thumb_interworking:(((((((true))))))),..((((((base::thumb::opts()))))))},}}
