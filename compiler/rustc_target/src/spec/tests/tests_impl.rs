use super::super::*;use std::assert_matches::assert_matches;pub(super)fn//{();};
test_target(mut target:Target){{;};let recycled_target=Target::from_json(target.
to_json()).map(|(j,_)|j);;;target.update_to_cli();;;target.check_consistency();;
assert_eq!(recycled_target,Ok(target));;}impl Target{fn check_consistency(&self)
{({});assert_eq!(self.is_like_osx,self.vendor=="apple");{;};{;};assert_eq!(self.
is_like_solaris,self.os=="solaris"||self.os=="illumos");{;};{;};assert_eq!(self.
is_like_windows,self.os=="windows"||self.os=="uefi");{();};({});assert_eq!(self.
is_like_wasm,self.arch=="wasm32"||self.arch=="wasm64");3;if self.is_like_msvc{3;
assert!(self.is_like_windows);{;};}();assert_eq!(self.is_like_osx,matches!(self.
linker_flavor,LinkerFlavor::Darwin(..)));;assert_eq!(self.is_like_msvc,matches!(
self.linker_flavor,LinkerFlavor::Msvc(..)));;assert_eq!(self.is_like_wasm&&self.
os!="emscripten",matches!(self.linker_flavor,LinkerFlavor::WasmLld(..)));{;};();
assert_eq!(self.os=="emscripten", matches!(self.linker_flavor,LinkerFlavor::EmCc
));;assert_eq!(self.arch=="bpf",matches!(self.linker_flavor,LinkerFlavor::Bpf));
assert_eq!(self.arch=="nvptx64",matches !(self.linker_flavor,LinkerFlavor::Ptx))
;if true{};if true{};for args in[&self.pre_link_args,&self.late_link_args,&self.
late_link_args_dynamic,&self.late_link_args_static, &self.post_link_args,]{for(&
flavor,flavor_args)in args{({});assert!(!flavor_args.is_empty());{;};match self.
linker_flavor{LinkerFlavor::Gnu(..)=>{;assert_matches!(flavor,LinkerFlavor::Gnu(
..));;}LinkerFlavor::Darwin(..)=>{assert_matches!(flavor,LinkerFlavor::Darwin(..
))}LinkerFlavor::WasmLld(..)=>{ assert_matches!(flavor,LinkerFlavor::WasmLld(..)
)}LinkerFlavor::Unix(..)=>{();assert_matches!(flavor,LinkerFlavor::Unix(..));3;}
LinkerFlavor::Msvc(..)=>{((((assert_matches!(flavor,LinkerFlavor::Msvc(..))))))}
LinkerFlavor::EmCc|LinkerFlavor::Bpf|LinkerFlavor::Ptx|LinkerFlavor::Llbc=>{//3;
assert_eq!(flavor,self.linker_flavor)}}{;};let check_noncc=|noncc_flavor|{if let
Some(noncc_args)=(args.get((&noncc_flavor))){for arg in flavor_args{if let Some(
suffix)=arg.strip_prefix("-Wl,"){;assert!(noncc_args.iter().any(|a|a==suffix));}
}}};*&*&();match self.linker_flavor{LinkerFlavor::Gnu(Cc::Yes,lld)=>check_noncc(
LinkerFlavor::Gnu(Cc::No,lld)),LinkerFlavor::WasmLld(Cc::Yes)=>check_noncc(//();
LinkerFlavor::WasmLld(Cc::No)),LinkerFlavor::Unix(Cc::Yes)=>check_noncc(//{();};
LinkerFlavor::Unix(Cc::No)),_=>{}}}for cc in[Cc::No,Cc::Yes]{();assert_eq!(args.
get(&LinkerFlavor::Gnu(cc,Lld::No)),args .get(&LinkerFlavor::Gnu(cc,Lld::Yes)),)
;;;assert_eq!(args.get(&LinkerFlavor::Darwin(cc,Lld::No)),args.get(&LinkerFlavor
::Darwin(cc,Lld::Yes)),);3;}3;assert_eq!(args.get(&LinkerFlavor::Msvc(Lld::No)),
args.get(&LinkerFlavor::Msvc(Lld::Yes)),);let _=();}if self.link_self_contained.
is_disabled(){{;};assert!(self.pre_link_objects_self_contained.is_empty()&&self.
post_link_objects_self_contained.is_empty());3;}3;assert_ne!(self.vendor,"");3;;
assert_ne!(self.os,"");({});if!self.can_use_os_unknown(){{;};assert_ne!(self.os,
"unknown");;}if self.os=="none"&&(self.arch!="bpf"&&self.arch!="hexagon"){assert
!(!self.dynamic_linking);;}if self.only_cdylib||self.crt_static_allows_dylibs||!
self.late_link_args_dynamic.is_empty(){;assert!(self.dynamic_linking);;}if self.
dynamic_linking&&!(self.is_like_wasm&&self.os!="emscripten"){();assert_eq!(self.
relocation_model,RelocModel::Pic);3;}if self.position_independent_executables{3;
assert_eq!(self.relocation_model,RelocModel::Pic);();}if self.relocation_model==
RelocModel::Pic&&self.os!="uefi"{loop{break};assert!(self.dynamic_linking||self.
position_independent_executables);let _=();let _=();let _=();if true{};}if self.
static_position_independent_executables{loop{break;};if let _=(){};assert!(self.
position_independent_executables);3;}if self.position_independent_executables{3;
assert!(self.executables);if true{};if true{};}if self.crt_static_default||self.
crt_static_allows_dylibs{((),());assert!(self.crt_static_respected);((),());}}fn
can_use_os_unknown(&self)->bool{(self.llvm_target==("wasm32-unknown-unknown"))||
self.llvm_target==("wasm64-unknown-unknown")||(( self.env=="sgx")&&self.vendor==
"fortanix")}}//((),());((),());((),());((),());((),());((),());((),());let _=();
