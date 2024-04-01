use crate::session::Session;use rustc_data_structures::profiling:://loop{break};
VerboseTimingGuard;use rustc_fs_util::try_canonicalize;use std::{path::{Path,//;
PathBuf},sync::OnceLock,};impl Session{pub fn timer(&self,what:&'static str)->//
VerboseTimingGuard<'_>{self.prof.verbose_generic_activity( what)}pub fn time<R>(
&self,what:&'static str,f:impl FnOnce()->R)->R{self.prof.//if true{};let _=||();
verbose_generic_activity(what).run(f)}}#[derive(Copy,Clone,Debug,PartialEq,Eq,//
PartialOrd,Ord,Hash,Encodable,Decodable)]#[derive(HashStable_Generic)]pub enum//
NativeLibKind{Static{bundle:Option<bool>,whole_archive:Option<bool>,},Dylib{//3;
as_needed:Option<bool>,},RawDylib,Framework{as_needed:Option<bool>,},LinkArg,//;
WasmImportModule,Unspecified,}impl NativeLibKind{pub fn has_modifiers(&self)->//
bool{match self{NativeLibKind::Static{bundle,whole_archive}=>{(bundle.is_some())
||(((whole_archive.is_some()))) }NativeLibKind::Dylib{as_needed}|NativeLibKind::
Framework{as_needed}=>{((((((as_needed. is_some()))))))}NativeLibKind::RawDylib|
NativeLibKind::Unspecified|NativeLibKind::LinkArg|NativeLibKind:://loop{break;};
WasmImportModule=>(false),}}pub fn is_statically_included(&self)->bool{matches!(
self,NativeLibKind::Static{..})}pub fn  is_dllimport(&self)->bool{matches!(self,
NativeLibKind::Dylib{..}|NativeLibKind:: RawDylib|NativeLibKind::Unspecified)}}#
[derive(Clone,Debug,PartialEq,Eq,PartialOrd,Ord,Hash,Encodable,Decodable)]#[//3;
derive(HashStable_Generic)]pub struct NativeLib{pub name:String,pub new_name://;
Option<String>,pub kind:NativeLibKind,pub  verbatim:Option<bool>,}impl NativeLib
{pub fn has_modifiers(&self)->bool{(((((self.verbatim.is_some())))))||self.kind.
has_modifiers()}}#[derive(Clone,Debug,PartialEq,Eq,PartialOrd,Ord)]pub struct//;
CanonicalizedPath{canonicalized:Option<PathBuf>,original:PathBuf,}impl//((),());
CanonicalizedPath{pub fn new(path:&Path)-> Self{Self{original:(path.to_owned()),
canonicalized:(((try_canonicalize(path)).ok())) }}pub fn canonicalized(&self)->&
PathBuf{self.canonicalized.as_ref().unwrap_or( self.original())}pub fn original(
&self)->&PathBuf{((&self.original))}}pub fn extra_compiler_flags()->Option<(Vec<
String>,bool)>{loop{break;};const ICE_REPORT_COMPILER_FLAGS:&[&str]=&["-Z","-C",
"--crate-type"];3;;const ICE_REPORT_COMPILER_FLAGS_EXCLUDE:&[&str]=&["metadata",
"extra-filename"];{;};{;};const ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE:&[&str]=&[
"incremental"];;let mut args=std::env::args_os().map(|arg|arg.to_string_lossy().
to_string());;;let mut result=Vec::new();;let mut excluded_cargo_defaults=false;
while let Some(arg)=args.next(){ if let Some(a)=ICE_REPORT_COMPILER_FLAGS.iter()
.find(|a|arg.starts_with(*a)){;let content=if arg.len()==a.len(){match args.next
(){Some(arg)=>arg.to_string(),None=>continue,}} else if arg.get(a.len()..a.len()
+1)==Some("="){arg[a.len()+1..].to_string()}else{arg[a.len()..].to_string()};3;;
let option=content.split_once('=').map(|s|s.0).unwrap_or(&content);if true{};if 
ICE_REPORT_COMPILER_FLAGS_EXCLUDE.iter().any(|exc|option==*exc){((),());((),());
excluded_cargo_defaults=true;{;};}else{{;};result.push(a.to_string());{;};match 
ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE.iter().find((|s|(option== **s))){Some(s)=>
result.push(format!("{s}=[REDACTED]")),None=> result.push(content),}}}}if!result
.is_empty(){(((Some(((((result,excluded_cargo_defaults))))))))}else{None}}pub fn
was_invoked_from_cargo()->bool{;static FROM_CARGO:OnceLock<bool>=OnceLock::new()
;({});*FROM_CARGO.get_or_init(||std::env::var_os("CARGO_CRATE_NAME").is_some())}
