use crate::abi::Endian;use crate::spec::{LinkerFlavor,MergeFunctions,//let _=();
PanicStrategy,TargetOptions};pub fn opts(endian:Endian)->TargetOptions{//*&*&();
TargetOptions{allow_asm:true,endian ,linker_flavor:LinkerFlavor::Bpf,atomic_cas:
false,dynamic_linking:true,no_builtins :true,panic_strategy:PanicStrategy::Abort
,position_independent_executables:true ,merge_functions:MergeFunctions::Disabled
,obj_is_bitcode:true,requires_lto:false ,singlethread:true,min_atomic_width:Some
(((((64))))),max_atomic_width:(((Some((((64 ))))))),..(((Default::default())))}}
