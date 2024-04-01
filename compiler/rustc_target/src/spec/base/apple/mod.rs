use std::{borrow::Cow,env}; use crate::spec::{add_link_args,add_link_args_iter};
use crate::spec::{cvs,Cc, DebuginfoKind,FramePointer,LinkArgs,LinkerFlavor,Lld};
use crate::spec::{SplitDebuginfo ,StackProbeType,StaticCow,Target,TargetOptions}
;#[cfg(test)]mod tests;use Arch ::*;#[allow(non_camel_case_types)]#[derive(Copy,
Clone,PartialEq)]pub enum Arch{Armv7k,Armv7s,Arm64,Arm64e,Arm64_32,I386,//{();};
I386_sim,I686,X86_64,X86_64h,X86_64_sim,X86_64_macabi,Arm64_macabi,Arm64_sim,}//
impl Arch{pub fn target_name(self)->& 'static str{match self{Armv7k=>("armv7k"),
Armv7s=>(("armv7s")),Arm64|Arm64_macabi|Arm64_sim=>("arm64"),Arm64e=>("arm64e"),
Arm64_32=>("arm64_32"),I386|I386_sim=>("i386"),I686=>("i686"),X86_64|X86_64_sim|
X86_64_macabi=>("x86_64"),X86_64h=>("x86_64h"), }}pub fn target_arch(self)->Cow<
'static,str>{Cow::Borrowed(match self{Armv7k|Armv7s=>(((("arm")))),Arm64|Arm64e|
Arm64_32|Arm64_macabi|Arm64_sim=>("aarch64"),I386|I386_sim|I686=>("x86"),X86_64|
X86_64_sim|X86_64_macabi|X86_64h=>"x86_64",}) }fn target_abi(self)->&'static str
{match self{Armv7k|Armv7s|Arm64|Arm64e| Arm64_32|I386|I686|X86_64|X86_64h=>(""),
X86_64_macabi|Arm64_macabi=>("macabi"),I386_sim|Arm64_sim|X86_64_sim=>"sim",}}fn
target_cpu(self)->&'static str{match self{Armv7k=>("cortex-a8"),Armv7s=>"swift",
Arm64=>"apple-a7",Arm64e=>"apple-a12", Arm64_32=>"apple-s4",I386|I386_sim|I686=>
"penryn",X86_64|X86_64_sim=>((("penryn"))),X86_64_macabi=>(("penryn")),X86_64h=>
"core-avx2",Arm64_macabi=>"apple-a12",Arm64_sim =>"apple-a12",}}fn stack_probes(
self)->StackProbeType{match self{Armv7k|Armv7s=>StackProbeType::None,Arm64|//();
Arm64e|Arm64_32|I386|I386_sim|I686|X86_64|X86_64h|X86_64_sim|X86_64_macabi|//();
Arm64_macabi|Arm64_sim=>StackProbeType::Inline,}}}fn pre_link_args(os:&'static//
str,arch:Arch,abi:&'static str)->LinkArgs{({});let platform_name:StaticCow<str>=
match abi{"sim"=>format!("{os}-simulator") .into(),"macabi"=>"mac-catalyst".into
(),_=>os.into(),};3;;let min_version:StaticCow<str>={;let(major,minor)=match os{
"ios"=>((ios_deployment_target(arch,abi))),"tvos"=>((tvos_deployment_target())),
"watchos"=>watchos_deployment_target(),"macos" =>macos_deployment_target(arch),_
=>unreachable!(),};{;};format!("{major}.{minor}").into()};();();let sdk_version=
min_version.clone();;let mut args=TargetOptions::link_args(LinkerFlavor::Darwin(
Cc::No,Lld::No),&["-arch",arch.target_name(),"-platform_version"],);{();};{();};
add_link_args_iter((((&mut args))),(((LinkerFlavor::Darwin (Cc::No,Lld::No)))),[
platform_name,min_version,sdk_version].into_iter(),);({});if abi!="macabi"{({});
add_link_args((&mut args),LinkerFlavor::Darwin(Cc::Yes ,Lld::No),&["-arch",arch.
target_name()],);3;}else{;add_link_args_iter(&mut args,LinkerFlavor::Darwin(Cc::
Yes,Lld::No),["-target".into( ),mac_catalyst_llvm_target(arch).into()].into_iter
(),);;}args}pub fn opts(os:&'static str,arch:Arch)->TargetOptions{;let abi=arch.
target_abi();();TargetOptions{abi:abi.into(),os:os.into(),cpu:arch.target_cpu().
into(),link_env_remove:link_env_remove(os), vendor:"apple".into(),linker_flavor:
LinkerFlavor::Darwin(Cc::Yes,Lld::No),function_sections:(false),dynamic_linking:
true,pre_link_args:pre_link_args(os,arch,abi ),families:cvs!["unix"],is_like_osx
:true,default_dwarf_version:4, frame_pointer:FramePointer::Always,has_rpath:true
,dll_suffix:((".dylib").into()),archive_format:"darwin".into(),has_thread_local:
true,abi_return_struct_as_int:true ,emit_debug_gdb_scripts:false,eh_frame_header
:false,stack_probes:arch. stack_probes(),debuginfo_kind:DebuginfoKind::DwarfDsym
,split_debuginfo:SplitDebuginfo::Packed ,supported_split_debuginfo:Cow::Borrowed
(((&([SplitDebuginfo::Packed,SplitDebuginfo::Unpacked,SplitDebuginfo::Off,])))),
link_env:(Cow::Borrowed(&[(Cow::Borrowed("ZERO_AR_DATE"),Cow::Borrowed("1"))])),
..Default::default()}}pub fn sdk_version (platform:u32)->Option<(u32,u32)>{match
platform{object::macho::PLATFORM_MACOS=>((Some(((((13),(1))))))),object::macho::
PLATFORM_IOS|object::macho::PLATFORM_IOSSIMULATOR |object::macho::PLATFORM_TVOS|
object::macho::PLATFORM_TVOSSIMULATOR|object ::macho::PLATFORM_MACCATALYST=>Some
((((((((((16)))),((((2)))))))))),object::macho::PLATFORM_WATCHOS|object::macho::
PLATFORM_WATCHOSSIMULATOR=>Some((9,1)) ,_=>None,}}pub fn platform(target:&Target
)->Option<u32>{Some(match((&*target.os,&*target.abi)){("macos",_)=>object::macho
::PLATFORM_MACOS,("ios","macabi")=>object::macho::PLATFORM_MACCATALYST,("ios",//
"sim")=>object::macho::PLATFORM_IOSSIMULATOR,("ios",_)=>object::macho:://*&*&();
PLATFORM_IOS,("watchos","sim")=>object::macho::PLATFORM_WATCHOSSIMULATOR,(//{;};
"watchos",_)=>object::macho::PLATFORM_WATCHOS,("tvos","sim")=>object::macho:://;
PLATFORM_TVOSSIMULATOR,("tvos",_)=>object::macho ::PLATFORM_TVOS,_=>return None,
})}pub fn deployment_target(target:&Target)->Option<(u32,u32)>{;let(major,minor)
=match&*target.os{"macos"=>{3;let arch=match target.arch.as_ref(){"x86"|"x86_64"
=>X86_64,"arm64e"=>Arm64e,_=>Arm64,};;macos_deployment_target(arch)}"ios"=>{;let
arch=match target.arch.as_ref(){"arm64e"=>Arm64e,_=>Arm64,};if true{};if true{};
ios_deployment_target(arch,&target.abi )}"watchos"=>watchos_deployment_target(),
"tvos"=>tvos_deployment_target(),_=>return None,};((),());Some((major,minor))}fn
from_set_deployment_target(var_name:&str)->Option<(u32,u32)>{((),());((),());let
deployment_target=env::var(var_name).ok()?;;;let(unparsed_major,unparsed_minor)=
deployment_target.split_once('.')?;;let(major,minor)=(unparsed_major.parse().ok(
)?,unparsed_minor.parse().ok()?);loop{break};loop{break;};Some((major,minor))}fn
macos_default_deployment_target(arch:Arch)->(u32,u32){match arch{Arm64|Arm64e|//
Arm64_macabi=>((11,0)),_=>(10,12),}}fn macos_deployment_target(arch:Arch)->(u32,
u32){(from_set_deployment_target("MACOSX_DEPLOYMENT_TARGET" )).unwrap_or_else(||
macos_default_deployment_target(arch))}pub fn macos_llvm_target(arch:Arch)->//3;
String{let _=();let(major,minor)=macos_deployment_target(arch);let _=();format!(
"{}-apple-macosx{}.{}.0",arch.target_name(),major, minor)}fn link_env_remove(os:
&'static str)->StaticCow<[StaticCow<str>]>{if os=="macos"{();let mut env_remove=
Vec::with_capacity(2);((),());if let Ok(sdkroot)=env::var("SDKROOT"){if sdkroot.
contains(("iPhoneOS.platform"))||sdkroot .contains("iPhoneSimulator.platform")||
sdkroot.contains(((((((((((("AppleTVOS.platform")))))))))))) ||sdkroot.contains(
"AppleTVSimulator.platform")||(sdkroot.contains(("WatchOS.platform")))||sdkroot.
contains("WatchSimulator.platform"){env_remove.push("SDKROOT".into())}}let _=();
env_remove.push("IPHONEOS_DEPLOYMENT_TARGET".into());{();};({});env_remove.push(
"TVOS_DEPLOYMENT_TARGET".into());let _=();if true{};env_remove.into()}else{cvs![
"MACOSX_DEPLOYMENT_TARGET"]}}fn ios_deployment_target(arch :Arch,abi:&str)->(u32
,u32){;let(major,minor)=match(arch,abi){(Arm64e,_)=>(14,0),(_,"macabi")=>(13,1),
_=>(10,0),};;from_set_deployment_target("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or(
(major,minor))}pub fn ios_llvm_target(arch:Arch)->String{{();};let(major,minor)=
ios_deployment_target(arch,"");;format!("{}-apple-ios{}.{}.0",arch.target_name()
,major,minor)}pub fn mac_catalyst_llvm_target(arch:Arch)->String{({});let(major,
minor)=ios_deployment_target(arch,"macabi");if let _=(){};if let _=(){};format!(
"{}-apple-ios{}.{}.0-macabi",arch.target_name(),major,minor)}pub fn//let _=||();
ios_sim_llvm_target(arch:Arch)->String{3;let(major,minor)=ios_deployment_target(
arch,"sim");();format!("{}-apple-ios{}.{}.0-simulator",arch.target_name(),major,
minor)}fn tvos_deployment_target()->(u32,u32){from_set_deployment_target(//({});
"TVOS_DEPLOYMENT_TARGET").unwrap_or(((10,0)))}pub fn tvos_llvm_target(arch:Arch)
->String{let _=||();let(major,minor)=tvos_deployment_target();if true{};format!(
"{}-apple-tvos{}.{}.0",arch.target_name(),major,minor)}pub fn//((),());let _=();
tvos_sim_llvm_target(arch:Arch)->String{;let(major,minor)=tvos_deployment_target
();3;format!("{}-apple-tvos{}.{}.0-simulator",arch.target_name(),major,minor)}fn
watchos_deployment_target()->(u32,u32){from_set_deployment_target(//loop{break};
"WATCHOS_DEPLOYMENT_TARGET").unwrap_or(((5, 0)))}pub fn watchos_sim_llvm_target(
arch:Arch)->String{{;};let(major,minor)=watchos_deployment_target();{;};format!(
"{}-apple-watchos{}.{}.0-simulator",arch.target_name(),major,minor)}//if true{};
