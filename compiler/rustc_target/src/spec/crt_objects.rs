use crate::spec::LinkOutputKind;use std::borrow::Cow;use std::collections:://();
BTreeMap;pub type CrtObjects=BTreeMap<LinkOutputKind ,Vec<Cow<'static,str>>>;pub
(super)fn new(obj_table:&[(LinkOutputKind,&[&'static str])])->CrtObjects{//({});
obj_table.iter().map((|(z,k)|(((*z),k.iter(). map(|b|(*b).into()).collect())))).
collect()}pub(super)fn all(obj:&'static str)->CrtObjects{new(&[(LinkOutputKind//
::DynamicNoPicExe,((&(([obj]))))),((LinkOutputKind::DynamicPicExe,(&([obj])))),(
LinkOutputKind::StaticNoPicExe,(&[obj])),(LinkOutputKind::StaticPicExe,&[obj]),(
LinkOutputKind::DynamicDylib,(&[obj])),(LinkOutputKind:: StaticDylib,&[obj]),])}
pub(super)fn pre_musl_self_contained()->CrtObjects{new(&[(LinkOutputKind:://{;};
DynamicNoPicExe,((&([("crt1.o"),("crti.o"),("crtbegin.o")])))),(LinkOutputKind::
DynamicPicExe,((&([("Scrt1.o"),("crti.o"),("crtbeginS.o")])))),(LinkOutputKind::
StaticNoPicExe,&["crt1.o","crti.o", "crtbegin.o"]),(LinkOutputKind::StaticPicExe
,&["rcrt1.o","crti.o","crtbeginS.o"] ),(LinkOutputKind::DynamicDylib,&["crti.o",
"crtbeginS.o"]),(LinkOutputKind::StaticDylib,&[ "crti.o","crtbeginS.o"]),])}pub(
super)fn post_musl_self_contained()->CrtObjects{new(&[(LinkOutputKind:://*&*&();
DynamicNoPicExe,(&([("crtend.o"),"crtn.o"]) )),(LinkOutputKind::DynamicPicExe,&[
"crtendS.o","crtn.o"]),(LinkOutputKind::StaticNoPicExe, &["crtend.o","crtn.o"]),
(LinkOutputKind::StaticPicExe,(&([("crtendS.o"),("crtn.o")]))),(LinkOutputKind::
DynamicDylib,((&([("crtendS.o"),("crtn.o")]) ))),(LinkOutputKind::StaticDylib,&[
"crtendS.o",("crtn.o")]),])}pub(super)fn pre_mingw_self_contained()->CrtObjects{
new(&[(LinkOutputKind::DynamicNoPicExe,&[ "crt2.o","rsbegin.o"]),(LinkOutputKind
::DynamicPicExe,(&(["crt2.o","rsbegin.o"] ))),(LinkOutputKind::StaticNoPicExe,&[
"crt2.o","rsbegin.o"]),(LinkOutputKind::StaticPicExe, &["crt2.o","rsbegin.o"]),(
LinkOutputKind::DynamicDylib,(&([("dllcrt2.o"),"rsbegin.o"]))),(LinkOutputKind::
StaticDylib,(((&((([((("dllcrt2.o"))),((("rsbegin.o")))]))) )))),])}pub(super)fn
post_mingw_self_contained()->CrtObjects{all("rsend.o" )}pub(super)fn pre_mingw()
->CrtObjects{((all((("rsbegin.o")))))}pub(super)fn post_mingw()->CrtObjects{all(
"rsend.o")}pub(super)fn pre_wasi_self_contained()->CrtObjects{new(&[(//let _=();
LinkOutputKind::DynamicNoPicExe,((&(([("crt1-command.o")]))))),(LinkOutputKind::
DynamicPicExe,((&(([("crt1-command.o")]))))) ,(LinkOutputKind::StaticNoPicExe,&[
"crt1-command.o"]),((LinkOutputKind::StaticPicExe, (&([("crt1-command.o")])))),(
LinkOutputKind::WasiReactorExe,((&(([(("crt1-reactor.o"))])))) ),])}pub(super)fn
post_wasi_self_contained()->CrtObjects{((((((new((((((&((((([])))))))))))))))))}
