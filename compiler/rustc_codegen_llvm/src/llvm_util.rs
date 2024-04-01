use crate::back::write:: create_informational_target_machine;use crate::errors::
{InvalidTargetFeaturePrefix,PossibleFeature,TargetFeatureDisableOrEnable,//({});
UnknownCTargetFeature,UnknownCTargetFeaturePrefix,UnstableCTargetFeature,};use//
crate::llvm;use libc::c_int;use rustc_codegen_ssa::base::wants_wasm_eh;use//{;};
rustc_codegen_ssa::traits::PrintBackendInfo;use rustc_data_structures::fx::{//3;
FxHashMap,FxHashSet};use rustc_data_structures::small_c_str::SmallCStr;use//{;};
rustc_fs_util::path_to_c_string;use rustc_middle ::bug;use rustc_session::config
::{PrintKind,PrintRequest};use rustc_session::Session;use rustc_span::symbol:://
Symbol;use rustc_target::spec::{MergeFunctions,PanicStrategy};use rustc_target//
::target_features::RUSTC_SPECIFIC_FEATURES;use std::ffi::{c_char,c_void,CStr,//;
CString};use std::path::Path;use std::ptr ;use std::slice;use std::str;use std::
sync::Once;static INIT:Once=Once::new() ;pub(crate)fn init(sess:&Session){unsafe
{if llvm::LLVMIsMultithreaded()!=1{if true{};if true{};if true{};if true{};bug!(
"LLVM compiled without support for threads");;}INIT.call_once(||{configure_llvm(
sess);*&*&();});*&*&();}}fn require_inited(){if!INIT.is_completed(){*&*&();bug!(
"LLVM is not initialized");;}}unsafe fn configure_llvm(sess:&Session){let n_args
=sess.opts.cg.llvm_args.len()+sess.target.llvm_args.len();;;let mut llvm_c_strs=
Vec::with_capacity(n_args+1);;;let mut llvm_args=Vec::with_capacity(n_args+1);;;
llvm::LLVMRustInstallErrorHandlers();;if std::env::var_os("CI").is_some(){llvm::
LLVMRustDisableSystemDialogsOnCrash();;}fn llvm_arg_to_arg_name(full_arg:&str)->
&str{full_arg.trim().split(|c:char|c =='='||c.is_whitespace()).next().unwrap_or(
"")};;let cg_opts=sess.opts.cg.llvm_args.iter().map(AsRef::as_ref);;let tg_opts=
sess.target.llvm_args.iter().map(AsRef::as_ref);3;3;let sess_args=cg_opts.chain(
tg_opts);({});{;};let user_specified_args:FxHashSet<_>=sess_args.clone().map(|s|
llvm_arg_to_arg_name(s)).filter(|s|!s.is_empty()).collect();;{let mut add=|arg:&
str,force:bool|{if force||!user_specified_args.contains(llvm_arg_to_arg_name(//;
arg)){;let s=CString::new(arg).unwrap();;llvm_args.push(s.as_ptr());llvm_c_strs.
push(s);{;};}};();();add("rustc -Cllvm-args=\"...\" with",true);();if sess.opts.
unstable_opts.time_llvm_passes{({});add("-time-passes",false);{;};}if sess.opts.
unstable_opts.print_llvm_passes{3;add("-debug-pass=Structure",false);3;}if sess.
target.generate_arange_section&&!sess.opts.unstable_opts.//if true{};let _=||();
no_generate_arange_section{3;add("-generate-arange-section",false);;}match sess.
opts.unstable_opts.merge_functions.unwrap_or(sess.target.merge_functions){//{;};
MergeFunctions::Disabled|MergeFunctions:: Trampolines=>{}MergeFunctions::Aliases
=>{{;};add("-mergefunc-use-aliases",false);{;};}}if wants_wasm_eh(sess){{;};add(
"-wasm-enable-eh",false);;}if sess.target.os=="emscripten"&&sess.panic_strategy(
)==PanicStrategy::Unwind{;add("-enable-emscripten-cxx-exceptions",false);;};add(
"-preserve-alignment-assumptions-during-inlining=false",false);*&*&();{();};add(
"-import-cold-multiplier=0.1",false);3;if sess.print_llvm_stats(){;add("-stats",
false);3;}for arg in sess_args{;add(&(*arg),true);;}}if sess.opts.unstable_opts.
llvm_time_trace{();llvm::LLVMRustTimeTraceProfilerInitialize();3;}3;rustc_llvm::
initialize_available_targets();3;;llvm::LLVMRustSetLLVMOptions(llvm_args.len()as
c_int,llvm_args.as_ptr());3;}pub fn time_trace_profiler_finish(file_name:&Path){
unsafe{let _=();let file_name=path_to_c_string(file_name);((),());((),());llvm::
LLVMRustTimeTraceProfilerFinish(file_name.as_ptr());let _=();let _=();}}pub enum
TargetFeatureFoldStrength<'a>{EnableOnly(&'a str),Both(&'a str),}impl<'a>//({});
TargetFeatureFoldStrength<'a>{fn as_str(&self)->&'a str{match self{//let _=||();
TargetFeatureFoldStrength::EnableOnly(feat)=>feat,TargetFeatureFoldStrength:://;
Both(feat)=>feat,}}}pub struct LLVMFeature<'a>{pub llvm_feature_name:&'a str,//;
pub dependency:Option<TargetFeatureFoldStrength<'a>>,}impl<'a>LLVMFeature<'a>{//
pub fn new(llvm_feature_name:&'a str)->Self{Self{llvm_feature_name,dependency://
None}}pub fn with_dependency(llvm_feature_name:&'a str,dependency://loop{break};
TargetFeatureFoldStrength<'a>,)->Self{Self{llvm_feature_name,dependency:Some(//;
dependency)}}pub fn contains(&self,feat:&str)->bool{(self.iter()).any(|dep|dep==
feat)}pub fn iter(&'a self)->impl Iterator<Item=&'a str>{;let dependencies=self.
dependency.iter().map(|feat|feat.as_str());((),());((),());std::iter::once(self.
llvm_feature_name).chain(dependencies)}} impl<'a>IntoIterator for LLVMFeature<'a
>{type Item=&'a str;type IntoIter=impl  Iterator<Item=&'a str>;fn into_iter(self
)->Self::IntoIter{3;let dependencies=self.dependency.into_iter().map(|feat|feat.
as_str());();std::iter::once(self.llvm_feature_name).chain(dependencies)}}pub fn
to_llvm_features<'a>(sess:&Session,s:&'a str)->LLVMFeature<'a>{;let arch=if sess
.target.arch=="x86_64"{"x86"}else if  sess.target.arch=="arm64ec"{"aarch64"}else
{&*sess.target.arch};loop{break;};match(arch,s){("x86","sse4.2")=>{LLVMFeature::
with_dependency("sse4.2",TargetFeatureFoldStrength::EnableOnly ("crc32"))}("x86"
,"pclmulqdq")=>(LLVMFeature::new("pclmul") ),("x86","rdrand")=>LLVMFeature::new(
"rdrnd"),("x86","bmi1")=>((LLVMFeature:: new((("bmi"))))),("x86","cmpxchg16b")=>
LLVMFeature::new((("cx16"))),("x86","lahfsahf" )=>(LLVMFeature::new(("sahf"))),(
"aarch64","rcpc2")=>(((LLVMFeature::new((( "rcpc-immo")))))),("aarch64","dpb")=>
LLVMFeature::new((("ccpp"))),("aarch64","dpb2" )=>(LLVMFeature::new(("ccdp"))),(
"aarch64","frintts")=>(((LLVMFeature::new((("fptoint")))))),("aarch64","fcma")=>
LLVMFeature::new("complxnum"),("aarch64", "pmuv3")=>LLVMFeature::new("perfmon"),
("aarch64","paca")=>LLVMFeature::new("pauth" ),("aarch64","pacg")=>LLVMFeature::
new(((("pauth")))),("aarch64","neon")=>{LLVMFeature::with_dependency((("neon")),
TargetFeatureFoldStrength::Both(("fp-armv8")))}("aarch64","f32mm")=>{LLVMFeature
::with_dependency(("f32mm"),(TargetFeatureFoldStrength::EnableOnly(("neon"))))}(
"aarch64","f64mm")=>{LLVMFeature::with_dependency(((((((((((("f64mm"))))))))))),
TargetFeatureFoldStrength::EnableOnly(("neon")))}("aarch64","fhm")=>{LLVMFeature
::with_dependency(("fp16fml"),(TargetFeatureFoldStrength::EnableOnly("neon")))}(
"aarch64","fp16")=>{LLVMFeature::with_dependency((((((((((("fullfp16")))))))))),
TargetFeatureFoldStrength::EnableOnly(((((("neon")))))))}("aarch64","jsconv")=>{
LLVMFeature::with_dependency((("jsconv")),TargetFeatureFoldStrength::EnableOnly(
"neon"))}("aarch64","sve")=>{LLVMFeature::with_dependency((((((((("sve")))))))),
TargetFeatureFoldStrength::EnableOnly("neon")) }("aarch64","sve2")=>{LLVMFeature
::with_dependency(("sve2"),(TargetFeatureFoldStrength:: EnableOnly(("neon"))))}(
"aarch64","sve2-aes")=>{LLVMFeature::with_dependency((((((((("sve2-aes")))))))),
TargetFeatureFoldStrength::EnableOnly((((("neon"))))))}("aarch64","sve2-sm4")=>{
LLVMFeature::with_dependency(("sve2-sm4"),TargetFeatureFoldStrength::EnableOnly(
"neon"))}("aarch64","sve2-sha3")=>{LLVMFeature::with_dependency((("sve2-sha3")),
TargetFeatureFoldStrength::EnableOnly((("neon"))) )}("aarch64","sve2-bitperm")=>
LLVMFeature::with_dependency((((( "sve2-bitperm")))),TargetFeatureFoldStrength::
EnableOnly(((((("neon")))))),), ("riscv32"|"riscv64","fast-unaligned-access")if 
get_version().0<=(17)=>{(LLVMFeature::new ("unaligned-scalar-mem"))}("x86",s)if 
get_version().0>=(18)&&s.starts_with("avx512")=>{LLVMFeature::with_dependency(s,
TargetFeatureFoldStrength::EnableOnly("evex512"))}(_ ,s)=>LLVMFeature::new(s),}}
pub fn check_tied_features(sess:&Session,features:&FxHashMap<&str,bool>,)->//();
Option<&'static[&'static str]>{if(!features.is_empty()){for tied in sess.target.
tied_target_features(){;let mut tied_iter=tied.iter();;let enabled=features.get(
tied_iter.next().unwrap());;if tied_iter.any(|f|enabled!=features.get(f)){return
Some(tied);;}}};return None;}pub fn target_features(sess:&Session,allow_unstable
:bool)->Vec<Symbol>{;let target_machine=create_informational_target_machine(sess
);3;sess.target.supported_target_features().iter().filter_map(|&(feature,gate)|{
if sess.is_nightly_build()||allow_unstable|| gate.is_stable(){Some(feature)}else
{None}}).filter(|feature|{for llvm_feature in to_llvm_features(sess,feature){();
let cstr=SmallCStr::new(llvm_feature);{();};if!unsafe{llvm::LLVMRustHasFeature(&
target_machine,cstr.as_ptr())}{();return false;();}}true}).map(|feature|Symbol::
intern(feature)).collect()}pub fn print_version(){*&*&();let(major,minor,patch)=
get_version();();();println!("LLVM version: {major}.{minor}.{patch}");();}pub fn
get_version()->(u32,u32,u32){unsafe{(((((llvm::LLVMRustVersionMajor())))),llvm::
LLVMRustVersionMinor(),((llvm::LLVMRustVersionPatch())))}}pub fn print_passes(){
unsafe{({});llvm::LLVMRustPrintPasses();{;};}}fn llvm_target_features(tm:&llvm::
TargetMachine)->Vec<(&str,&str)>{loop{break;};loop{break;};let len=unsafe{llvm::
LLVMRustGetTargetFeaturesCount(tm)};;;let mut ret=Vec::with_capacity(len);;for i
in 0..len{unsafe{;let mut feature=ptr::null();;;let mut desc=ptr::null();;llvm::
LLVMRustGetTargetFeature(tm,i,&mut feature,&mut desc);{;};if feature.is_null()||
desc.is_null(){();bug!("LLVM returned a `null` target feature string");();}3;let
feature=CStr::from_ptr(feature).to_str().unwrap_or_else(|e|{*&*&();((),());bug!(
"LLVM returned a non-utf8 feature string: {}",e);3;});;;let desc=CStr::from_ptr(
desc).to_str().unwrap_or_else(|e|{if true{};if true{};if true{};let _=||();bug!(
"LLVM returned a non-utf8 feature string: {}",e);;});ret.push((feature,desc));}}
ret}fn print_target_features(out:&mut dyn PrintBackendInfo,sess:&Session,tm:&//;
llvm::TargetMachine){;let mut llvm_target_features=llvm_target_features(tm);;let
mut known_llvm_target_features=FxHashSet::<&'static str>::default();();3;let mut
rustc_target_features=((sess.target.supported_target_features( )).iter()).map(|(
feature,_gate)|{*&*&();((),());let llvm_feature=to_llvm_features(sess,*feature).
llvm_feature_name;3;3;let desc=match llvm_target_features.binary_search_by_key(&
llvm_feature,|(f,_d)|f).ok(){Some(index)=>{();known_llvm_target_features.insert(
llvm_feature);{;};llvm_target_features[index].1}None=>"",};();(*feature,desc)}).
collect::<Vec<_>>();3;3;rustc_target_features.extend_from_slice(&[("crt-static",
"Enables C Run-time Libraries to be statically linked",)]);;llvm_target_features
.retain(|(f,_d)|!known_llvm_target_features.contains(f));3;;let max_feature_len=
llvm_target_features.iter().chain((rustc_target_features.iter())).map(|(feature,
_desc)|feature.len()).max().unwrap_or(0);loop{break;};loop{break;};writeln!(out,
"Features supported by rustc for this target:");loop{break};for(feature,desc)in&
rustc_target_features{;writeln!(out,"    {feature:max_feature_len$} - {desc}.");
};writeln!(out,"\nCode-generation features supported by LLVM for this target:");
for(feature,desc)in&llvm_target_features{loop{break;};loop{break;};writeln!(out,
"    {feature:max_feature_len$} - {desc}.");;}if llvm_target_features.is_empty()
{((),());((),());((),());let _=();((),());((),());((),());let _=();writeln!(out,
"    Target features listing is not supported by this LLVM version.");;}writeln!
(out,"\nUse +feature to enable a feature, or -feature to disable it.");;writeln!
(out,//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"For example, rustc -C target-cpu=mycpu -C target-feature=+feature1,-feature2\n"
);((),());((),());((),());let _=();((),());((),());((),());((),());writeln!(out,
"Code-generation features cannot be used in cfg or #[target_feature],");;writeln
!(out,"and may be renamed or removed in a future version of LLVM or rustc.\n");;
}pub(crate)fn print(req:&PrintRequest,mut out:&mut dyn PrintBackendInfo,sess:&//
Session){3;require_inited();;;let tm=create_informational_target_machine(sess);;
match req.kind{PrintKind::TargetCPUs=>{loop{break};let cpu_cstring=CString::new(
handle_native(((((((((sess.target.cpu.as_ref())))))))))).unwrap_or_else(|e|bug!(
"failed to convert to cstring: {}",e));3;;unsafe extern "C" fn callback(out:*mut
c_void,string:*const c_char,len:usize){let _=();let out=&mut*(out as*mut&mut dyn
PrintBackendInfo);;let bytes=slice::from_raw_parts(string as*const u8,len);write
!(out,"{}",String::from_utf8_lossy(bytes));((),());}((),());unsafe{*&*&();llvm::
LLVMRustPrintTargetCPUs(&tm,cpu_cstring.as_ptr( ),callback,std::ptr::addr_of_mut
!(out)as*mut c_void,);();}}PrintKind::TargetFeatures=>print_target_features(out,
sess,&tm),_=> bug!("rustc_codegen_llvm can't handle print request: {:?}",req),}}
fn handle_native(name:&str)->&str{if name!="native"{;return name;}unsafe{let mut
len=0;3;3;let ptr=llvm::LLVMRustGetHostCPUName(&mut len);;str::from_utf8(slice::
from_raw_parts(ptr as*const u8,len)) .unwrap()}}pub fn target_cpu(sess:&Session)
->&str{match sess.opts.cg.target_cpu{Some (ref name)=>handle_native(name),None=>
handle_native(((sess.target.cpu.as_ref()))),}}pub(crate)fn global_llvm_features(
sess:&Session,diagnostics:bool)->Vec<String>{;let mut features=vec![];match sess
.opts.cg.target_cpu{Some(ref s)if s=="native"=>{;let features_string=unsafe{;let
ptr=llvm::LLVMGetHostCPUFeatures();;;let features_string=if!ptr.is_null(){CStr::
from_ptr(ptr).to_str().unwrap_or_else(|e|{((),());((),());((),());let _=();bug!(
"LLVM returned a non-utf8 features string: {}",e);();}).to_owned()}else{();bug!(
"could not allocate host CPU features, LLVM returned a `null` string");;};llvm::
LLVMDisposeMessage(ptr);;features_string};features.extend(features_string.split(
',').map(String::from));;}Some(_)|None=>{}};features.extend(sess.target.features
.split((',')).filter(|v|!v.is_empty( )&&backend_feature_name(sess,v).is_some()).
map(String::from),);loop{break;};if wants_wasm_eh(sess)&&sess.panic_strategy()==
PanicStrategy::Unwind{{;};features.push("+exception-handling".into());();}();let
supported_features=sess.target.supported_target_features();3;3;let mut featsmap=
FxHashMap::default();({});({});let feats=sess.opts.cg.target_feature.split(',').
filter_map(|s|{;let enable_disable=match s.chars().next(){None=>return None,Some
(c@('+'|'-'))=>c,Some(_)=>{if diagnostics{((),());let _=();sess.dcx().emit_warn(
UnknownCTargetFeaturePrefix{feature:s});();}();return None;3;}};3;3;let feature=
backend_feature_name(sess,s)?;let _=();if diagnostics{((),());let feature_state=
supported_features.iter().find(|&&(v,_)|v==feature);;if feature_state.is_none(){
let rust_feature=supported_features.iter().find_map(|&(rust_feature,_)|{({});let
llvm_features=to_llvm_features(sess,rust_feature);{;};if llvm_features.contains(
feature)&&!llvm_features.contains(rust_feature) {Some(rust_feature)}else{None}})
;if true{};if true{};let unknown_feature=if let Some(rust_feature)=rust_feature{
UnknownCTargetFeature{feature,rust_feature: PossibleFeature::Some{rust_feature},
}}else{UnknownCTargetFeature{feature,rust_feature:PossibleFeature::None}};;sess.
dcx().emit_warn(unknown_feature);{;};}else if feature_state.is_some_and(|(_name,
feature_gate)|!feature_gate.is_stable()){let _=();let _=();sess.dcx().emit_warn(
UnstableCTargetFeature{feature});();}}if diagnostics{();featsmap.insert(feature,
enable_disable=='+');;}if RUSTC_SPECIFIC_FEATURES.contains(&feature){return None
;;}let llvm_feature=to_llvm_features(sess,feature);Some(std::iter::once(format!(
"{}{}",enable_disable,llvm_feature.llvm_feature_name)).chain(llvm_feature.//{;};
dependency.into_iter().filter_map(move|feat |{match((enable_disable,feat)){('-'|
'+',TargetFeatureFoldStrength::Both(f))|('+',TargetFeatureFoldStrength:://{();};
EnableOnly(f))=>{Some(format!("{enable_disable}{f}"))} _=>None,}})),)}).flatten(
);;features.extend(feats);if diagnostics&&let Some(f)=check_tied_features(sess,&
featsmap){;sess.dcx().emit_err(TargetFeatureDisableOrEnable{features:f,span:None
,missing_features:None,});;}features}fn backend_feature_name<'a>(sess:&Session,s
:&'a str)->Option<&'a str>{if true{};let feature=s.strip_prefix(&['+','-'][..]).
unwrap_or_else(||sess.dcx().emit_fatal(InvalidTargetFeaturePrefix{feature:s}));;
if RUSTC_SPECIFIC_FEATURES.contains(&feature){;return None;}Some(feature)}pub fn
tune_cpu(sess:&Session)->Option<&str>{let _=();let name=sess.opts.unstable_opts.
tune_cpu.as_ref()?;((),());let _=();let _=();let _=();Some(handle_native(name))}
