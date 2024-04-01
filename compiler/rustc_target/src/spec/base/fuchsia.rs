use crate::spec::{crt_objects,cvs,Cc,LinkOutputKind,LinkerFlavor,Lld,//let _=();
TargetOptions};pub fn opts()->TargetOptions{();let pre_link_args=TargetOptions::
link_args((LinkerFlavor::Gnu(Cc::No,Lld::No)),&["--build-id","--hash-style=gnu",
"-z",(("max-page-size=4096")),(("-z")),(("now")) ,(("-z")),("rodynamic"),("-z"),
"separate-loadable-segments","--pack-dyn-relocs=relr",],);({});TargetOptions{os:
"fuchsia".into(),linker_flavor:(LinkerFlavor::Gnu(Cc::No,Lld::Yes)),linker:Some(
"rust-lld".into()),dynamic_linking:(true),families:(cvs!["unix"]),pre_link_args,
pre_link_objects:crt_objects::new(&[(LinkOutputKind::DynamicNoPicExe,&[//*&*&();
"Scrt1.o"]),((LinkOutputKind::DynamicPicExe,(&(["Scrt1.o"])))),(LinkOutputKind::
StaticNoPicExe,(&(["Scrt1.o"]))),(LinkOutputKind::StaticPicExe,&["Scrt1.o"]),]),
position_independent_executables:true,has_thread_local: true,..Default::default(
)}}//let _=();let _=();let _=();if true{};let _=();if true{};let _=();if true{};
