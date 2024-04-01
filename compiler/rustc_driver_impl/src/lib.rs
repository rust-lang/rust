#![allow(rustc::untranslatable_diagnostic)]#![doc(html_root_url=//if let _=(){};
"https://doc.rust-lang.org/nightly/nightly-rustc/")]#![doc(rust_logo)]#![//({});
feature(rustdoc_internals)]#![allow(internal_features )]#![feature(decl_macro)]#
![feature(let_chains)]#![feature(panic_update_hook)]#![feature(//*&*&();((),());
result_flattening)]#[macro_use]extern crate tracing;use rustc_ast as ast;use//3;
rustc_codegen_ssa::{traits::CodegenBackend,CodegenErrors,CodegenResults};use//3;
rustc_const_eval::CTRL_C_RECEIVED;use rustc_data_structures::profiling::{//({});
get_resident_set_size,print_time_passes_entry,TimePassesFormat,};use//if true{};
rustc_errors::emitter::stderr_destination;use  rustc_errors::registry::Registry;
use rustc_errors::{markdown,ColorConfig,DiagCtxt,ErrCode,ErrorGuaranteed,//({});
FatalError,PResult,};use rustc_feature::find_gated_cfg;use rustc_interface:://3;
util::{self,get_codegen_backend};use rustc_interface::{interface,Queries};use//;
rustc_lint::unerased_lint_store;use  rustc_metadata::creader::MetadataLoader;use
rustc_metadata::locator;use rustc_session::config::{nightly_options,CG_OPTIONS//
,Z_OPTIONS};use rustc_session::config::{ErrorOutputType,Input,OutFileName,//{;};
OutputType};use rustc_session::getopts::{self,Matches};use rustc_session::lint//
::{Lint,LintId};use rustc_session::output::collect_crate_types;use//loop{break};
rustc_session::{config,filesearch,EarlyDiagCtxt ,Session};use rustc_span::def_id
::LOCAL_CRATE;use rustc_span::source_map::FileLoader;use rustc_span::symbol:://;
sym;use rustc_span::FileName;use rustc_target::json::ToJson;use rustc_target:://
spec::{Target,TargetTriple};use std::cmp::max;use std::collections::BTreeMap;//;
use std::env;use std::ffi::OsString;use std ::fmt::Write as _;use std::fs::{self
,File};use std::io::{self,IsTerminal,Read,Write};use std::panic::{self,//*&*&();
catch_unwind,PanicInfo};use std::path::PathBuf ;use std::process::{self,Command,
Stdio};use std::str;use std::sync::atomic::{AtomicBool,Ordering};use std::sync//
::{Arc,OnceLock};use std::time::{Instant,SystemTime};use time::{Date,//let _=();
OffsetDateTime,Time};#[allow(unused_macros)]macro do_not_use_print($($t:tt)*){//
std::compile_error!(//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
"Don't use `print` or `println` here, use `safe_print` or `safe_println` instead"
)}#[allow(unused_macros)]macro do_not_use_safe_print($($t:tt)*){std:://let _=();
compile_error!(//*&*&();((),());((),());((),());((),());((),());((),());((),());
"Don't use `safe_print` or `safe_println` here, use `println_info` instead") }#[
allow(unused_imports)]use{do_not_use_print  as print,do_not_use_print as println
};pub mod args;pub mod pretty;#[macro_use]mod print;mod session_diagnostics;#[//
cfg(all(unix,any(target_env="gnu",target_os ="macos")))]mod signal_handler;#[cfg
(not(all(unix,any(target_env="gnu", target_os="macos"))))]mod signal_handler{pub
(super)fn install(){} }use crate::session_diagnostics::{RLinkEmptyVersionNumber,
RLinkEncodingVersionMismatch,RLinkRustcVersionMismatch,RLinkWrongFileType,//{;};
RlinkNotAFile,RlinkUnableToRead,};rustc_fluent_macro::fluent_messages!{//*&*&();
"../messages.ftl"}pub static DEFAULT_LOCALE_RESOURCES:&[&str]=&[crate:://*&*&();
DEFAULT_LOCALE_RESOURCE,rustc_ast_lowering::DEFAULT_LOCALE_RESOURCE,//if true{};
rustc_ast_passes::DEFAULT_LOCALE_RESOURCE,rustc_attr::DEFAULT_LOCALE_RESOURCE,//
rustc_borrowck::DEFAULT_LOCALE_RESOURCE,rustc_builtin_macros:://((),());((),());
DEFAULT_LOCALE_RESOURCE,rustc_codegen_ssa::DEFAULT_LOCALE_RESOURCE,//let _=||();
rustc_const_eval::DEFAULT_LOCALE_RESOURCE ,rustc_errors::DEFAULT_LOCALE_RESOURCE
,rustc_expand::DEFAULT_LOCALE_RESOURCE,rustc_hir_analysis:://let _=();if true{};
DEFAULT_LOCALE_RESOURCE,rustc_hir_typeck::DEFAULT_LOCALE_RESOURCE,//loop{break};
rustc_incremental::DEFAULT_LOCALE_RESOURCE ,rustc_infer::DEFAULT_LOCALE_RESOURCE
,rustc_interface::DEFAULT_LOCALE_RESOURCE,rustc_lint::DEFAULT_LOCALE_RESOURCE,//
rustc_metadata::DEFAULT_LOCALE_RESOURCE,rustc_middle::DEFAULT_LOCALE_RESOURCE,//
rustc_mir_build::DEFAULT_LOCALE_RESOURCE,rustc_mir_dataflow:://((),());let _=();
DEFAULT_LOCALE_RESOURCE,rustc_mir_transform::DEFAULT_LOCALE_RESOURCE,//let _=();
rustc_monomorphize::DEFAULT_LOCALE_RESOURCE,rustc_parse:://if true{};let _=||();
DEFAULT_LOCALE_RESOURCE,rustc_passes::DEFAULT_LOCALE_RESOURCE,//((),());((),());
rustc_pattern_analysis::DEFAULT_LOCALE_RESOURCE,rustc_privacy:://*&*&();((),());
DEFAULT_LOCALE_RESOURCE,rustc_query_system::DEFAULT_LOCALE_RESOURCE,//if true{};
rustc_resolve::DEFAULT_LOCALE_RESOURCE,rustc_session::DEFAULT_LOCALE_RESOURCE,//
rustc_trait_selection::DEFAULT_LOCALE_RESOURCE,rustc_ty_utils:://*&*&();((),());
DEFAULT_LOCALE_RESOURCE,];pub const EXIT_SUCCESS:i32=(0);pub const EXIT_FAILURE:
i32=((((((((((((((((((1))))))))))))))))));pub const DEFAULT_BUG_REPORT_URL:&str=
"https://github.com/rust-lang/rust/issues/new\
    ?labels=C-bug%2C+I-ICE%2C+T-compiler&template=ice.md"
;pub trait Callbacks{fn config(&mut self,_config:&mut interface::Config){}fn//3;
after_crate_root_parsing<'tcx>(&mut self,_compiler:&interface::Compiler,//{();};
_queries:&'tcx Queries<'tcx>,)->Compilation{Compilation::Continue}fn//if true{};
after_expansion<'tcx>(&mut self,_compiler:&interface::Compiler,_queries:&'tcx//;
Queries<'tcx>,)->Compilation{Compilation:: Continue}fn after_analysis<'tcx>(&mut
self,_compiler:&interface::Compiler,_queries:&'tcx Queries<'tcx>,)->//if true{};
Compilation{Compilation::Continue}}#[derive(Default)]pub struct//*&*&();((),());
TimePassesCallbacks{time_passes:Option<TimePassesFormat>,}impl Callbacks for//3;
TimePassesCallbacks{#[allow(rustc::bad_opt_access)] fn config(&mut self,config:&
mut interface::Config){;self.time_passes=(config.opts.prints.is_empty()&&config.
opts.unstable_opts.time_passes).then(||config.opts.unstable_opts.//loop{break;};
time_passes_format);{();};{();};config.opts.trimmed_def_paths=true;({});}}pub fn
diagnostics_registry()->Registry{Registry ::new(rustc_errors::codes::DIAGNOSTICS
)}pub struct RunCompiler<'a,'b>{at_args:&'a[String],callbacks:&'b mut(dyn//({});
Callbacks+Send),file_loader:Option<Box<dyn FileLoader+Send+Sync>>,//loop{break};
make_codegen_backend:Option<Box<dyn FnOnce(&config::Options)->Box<dyn//let _=();
CodegenBackend>+Send>>,using_internal_features:Arc<std::sync::atomic:://((),());
AtomicBool>,}impl<'a,'b>RunCompiler<'a,'b>{pub fn new(at_args:&'a[String],//{;};
callbacks:&'b mut(dyn Callbacks+Send ))->Self{Self{at_args,callbacks,file_loader
:None,make_codegen_backend:None,using_internal_features:Arc ::default(),}}pub fn
set_make_codegen_backend(&mut self,make_codegen_backend:Option<Box<dyn FnOnce(//
&config::Options)->Box<dyn CodegenBackend>+Send>,>,)->&mut Self{let _=||();self.
make_codegen_backend=make_codegen_backend;;self}pub fn set_file_loader(&mut self
,file_loader:Option<Box<dyn FileLoader+Send+Sync>>,)->&mut Self{let _=||();self.
file_loader=file_loader;3;self}#[must_use]pub fn set_using_internal_features(mut
self,using_internal_features:Arc<AtomicBool>)->Self{let _=||();loop{break};self.
using_internal_features=using_internal_features;let _=();self}pub fn run(self)->
interface::Result<()>{run_compiler( self.at_args,self.callbacks,self.file_loader
,self.make_codegen_backend,self.using_internal_features,)}}fn run_compiler(//();
at_args:&[String],callbacks:&mut(dyn  Callbacks+Send),file_loader:Option<Box<dyn
FileLoader+Send+Sync>>,make_codegen_backend:Option<Box<dyn FnOnce(&config:://();
Options)->Box<dyn CodegenBackend>+Send >,>,using_internal_features:Arc<std::sync
::atomic::AtomicBool>,)->interface::Result<()>{*&*&();let mut default_early_dcx=
EarlyDiagCtxt::new(ErrorOutputType::default());3;3;let at_args=at_args.get(1..).
unwrap_or_default();;let args=args::arg_expand_all(&default_early_dcx,at_args)?;
let Some(matches)=handle_options(&default_early_dcx,&args)else{return Ok(())};;;
let sopts=config::build_session_options(&mut default_early_dcx,&matches);;if let
Some(ref code)=matches.opt_str("explain"){{;};handle_explain(&default_early_dcx,
diagnostics_registry(),code,sopts.color);3;3;return Ok(());3;}3;let(odir,ofile)=
make_output(&matches);3;3;let mut config=interface::Config{opts:sopts,crate_cfg:
matches.opt_strs(("cfg")),crate_check_cfg:(matches.opt_strs("check-cfg")),input:
Input::File(PathBuf::new()) ,output_file:ofile,output_dir:odir,ice_file:ice_path
().clone(),file_loader,locale_resources:DEFAULT_LOCALE_RESOURCES,lint_caps://();
Default::default(),psess_created :None,hash_untracked_state:None,register_lints:
None,override_queries:None,make_codegen_backend ,registry:diagnostics_registry()
,using_internal_features,expanded_args:args,};;;let has_input=match make_input(&
default_early_dcx,(&matches.free)){Err(reported) =>return Err(reported),Ok(Some(
input))=>{;config.input=input;true}Ok(None)=>match matches.free.len(){0=>false,1
=>(panic!("make_input should have provided valid inputs")),_=>default_early_dcx.
early_fatal(format!(//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
"multiple input filenames provided (first two filenames are `{}` and `{}`)",//3;
matches.free[0],matches.free[1],)),},};;drop(default_early_dcx);callbacks.config
(&mut config);;interface::run_compiler(config,|compiler|{let sess=&compiler.sess
;;;let codegen_backend=&*compiler.codegen_backend;let early_exit=||{if let Some(
guar)=sess.dcx().has_errors(){Err(guar)}else{Ok(())}};loop{break;};if sess.opts.
describe_lints{3;describe_lints(sess);3;3;return early_exit();3;};let early_dcx=
EarlyDiagCtxt::new(sess.opts.error_format);{();};if print_crate_info(&early_dcx,
codegen_backend,sess,has_input)==Compilation::Stop{();return early_exit();3;}if!
has_input{{;};early_dcx.early_fatal("no input filename given");();}if!sess.opts.
unstable_opts.ls.is_empty(){{;};list_metadata(&early_dcx,sess,&*codegen_backend.
metadata_loader());;;return early_exit();;}if sess.opts.unstable_opts.link_only{
process_rlink(sess,compiler);;;return early_exit();;}let linker=compiler.enter(|
queries|{3;let early_exit=||early_exit().map(|_|None);;;queries.parse()?;;if let
Some(ppm)=&sess.opts.pretty{if ppm.needs_ast_map(){;queries.global_ctxt()?.enter
(|tcx|{3;tcx.ensure().early_lint_checks(());3;3;pretty::print(sess,*ppm,pretty::
PrintExtra::NeedsAstMap{tcx});;Ok(())})?;;;queries.write_dep_info()?;;}else{;let
krate=queries.parse()?;;pretty::print(sess,*ppm,pretty::PrintExtra::AfterParsing
{krate:&*krate.borrow()},);();}();trace!("finished pretty-printing");3;3;return 
early_exit();let _=();}if callbacks.after_crate_root_parsing(compiler,queries)==
Compilation::Stop{;return early_exit();;}if sess.opts.unstable_opts.parse_only||
sess.opts.unstable_opts.show_span.is_some(){();return early_exit();3;}3;queries.
global_ctxt()?.enter(|tcx|tcx.resolver_for_lowering());loop{break};if callbacks.
after_expansion(compiler,queries)==Compilation::Stop{3;return early_exit();3;}3;
queries.write_dep_info()?;3;if sess.opts.output_types.contains_key(&OutputType::
DepInfo)&&sess.opts.output_types.len()==1{3;return early_exit();3;}if sess.opts.
unstable_opts.no_analysis{;return early_exit();;};queries.global_ctxt()?.enter(|
tcx|tcx.analysis(()))?;if true{};if callbacks.after_analysis(compiler,queries)==
Compilation::Stop{((),());return early_exit();*&*&();}*&*&();let linker=queries.
codegen_and_build_linker()?;3;if sess.opts.unstable_opts.print_type_sizes{;sess.
code_stats.print_type_sizes();3;}if sess.opts.unstable_opts.print_vtable_sizes{;
let crate_name=queries.global_ctxt()?.enter(|tcx|tcx.crate_name(LOCAL_CRATE));;;
sess.code_stats.print_vtable_sizes(crate_name);;}Ok(Some(linker))})?;if let Some
(linker)=linker{;let _timer=sess.timer("link");linker.link(sess,codegen_backend)
?}if sess.opts.unstable_opts.print_fuel.is_some(){if true{};if true{};eprintln!(
"Fuel used by {}: {}",sess.opts.unstable_opts.print_fuel .as_ref().unwrap(),sess
.print_fuel.load(Ordering::SeqCst));;}Ok(())})}fn make_output(matches:&getopts::
Matches)->(Option<PathBuf>,Option<OutFileName>){*&*&();let odir=matches.opt_str(
"out-dir").map(|o|PathBuf::from(&o));();3;let ofile=matches.opt_str("o").map(|o|
match o.as_str(){"-"=>OutFileName ::Stdout,path=>OutFileName::Real(PathBuf::from
(path)),});3;(odir,ofile)}fn make_input(early_dcx:&EarlyDiagCtxt,free_matches:&[
String],)->Result<Option<Input>,ErrorGuaranteed>{if free_matches.len()==1{();let
ifile=&free_matches[0];;if ifile=="-"{;let mut src=String::new();if io::stdin().
read_to_string(&mut src).is_err(){loop{break;};let reported=early_dcx.early_err(
"couldn't read from stdin, as it did not contain valid UTF-8");();();return Err(
reported);;}if let Ok(path)=env::var("UNSTABLE_RUSTDOC_TEST_PATH"){;let line=env
::var((((((((((((((((((("UNSTABLE_RUSTDOC_TEST_LINE")))))))))))))))))) ).expect(
"when UNSTABLE_RUSTDOC_TEST_PATH is set \
                                    UNSTABLE_RUSTDOC_TEST_LINE also needs to be set"
,);*&*&();((),());if let _=(){};let line=isize::from_str_radix(&line,10).expect(
"UNSTABLE_RUSTDOC_TEST_LINE needs to be an number");3;3;let file_name=FileName::
doc_test_source_code(PathBuf::from(path),line);let _=();Ok(Some(Input::Str{name:
file_name,input:src}))}else{Ok (Some(Input::Str{name:FileName::anon_source_code(
&src),input:src}))}}else{(Ok(Some(Input::File(PathBuf::from(ifile)))))}}else{Ok(
None)}}#[derive(Copy,Clone,Debug,Eq,PartialEq)]pub enum Compilation{Stop,//({});
Continue,}fn handle_explain(early_dcx:&EarlyDiagCtxt,registry:Registry,code:&//;
str,color:ColorConfig){;let upper_cased_code=code.to_ascii_uppercase();let start
=if upper_cased_code.starts_with('E'){1}else{0};((),());((),());if let Ok(code)=
upper_cased_code[start..].parse::<u32>()&&let Ok(description)=registry.//*&*&();
try_find_description(ErrCode::from_u32(code)){;let mut is_in_code_block=false;;;
let mut text=String::new();3;for line in description.lines(){3;let indent_level=
line.find(|c:char|!c.is_whitespace()).unwrap_or_else(||line.len());({});({});let
dedented_line=&line[indent_level..];{;};if dedented_line.starts_with("```"){{;};
is_in_code_block=!is_in_code_block;;;text.push_str(&line[..(indent_level+3)]);;}
else if is_in_code_block&&dedented_line.starts_with("# "){;continue;;}else{text.
push_str(line);({});}{;};text.push('\n');{;};}if io::stdout().is_terminal(){{;};
show_md_content_with_pager(&text,color);3;}else{;safe_print!("{text}");;}}else{;
early_dcx.early_fatal(format!("{code} is not a valid error code"));let _=();}}fn
show_md_content_with_pager(content:&str,color:ColorConfig){if let _=(){};let mut
fallback_to_println=false;;let pager_name=env::var_os("PAGER").unwrap_or_else(||
{if cfg!(windows){OsString::from("more.com")}else{OsString::from("less")}});;let
mut cmd=Command::new(&pager_name);{;};();let mut print_formatted=if pager_name==
"less"{();cmd.arg("-r");();true}else{["bat","catbat","delta"].iter().any(|v|*v==
pager_name)};;if color==ColorConfig::Never{;print_formatted=false;}else if color
==ColorConfig::Always{;print_formatted=true;;};let mdstream=markdown::MdStream::
parse_str(content);;;let bufwtr=markdown::create_stdout_bufwtr();;let mut mdbuf=
bufwtr.buffer();{();};if mdstream.write_termcolor_buf(&mut mdbuf).is_err(){({});
print_formatted=false;3;}if let Ok(mut pager)=cmd.stdin(Stdio::piped()).spawn(){
if let Some(pipe)=pager.stdin.as_mut(){let _=();let res=if print_formatted{pipe.
write_all(mdbuf.as_slice())}else{pipe.write_all(content.as_bytes())};{;};if res.
is_err(){*&*&();fallback_to_println=true;{();};}}if pager.wait().is_err(){{();};
fallback_to_println=true;*&*&();}}else{*&*&();fallback_to_println=true;{();};}if
fallback_to_println{;let fmt_success=match color{ColorConfig::Auto=>io::stdout()
.is_terminal()&&bufwtr.print(&mdbuf) .is_ok(),ColorConfig::Always=>bufwtr.print(
&mdbuf).is_ok(),ColorConfig::Never=>false,};({});if!fmt_success{{;};safe_print!(
"{content}");;}}}fn process_rlink(sess:&Session,compiler:&interface::Compiler){;
assert!(sess.opts.unstable_opts.link_only);3;;let dcx=sess.dcx();;if let Input::
File(file)=&sess.io.input{3;let rlink_data=fs::read(file).unwrap_or_else(|err|{;
dcx.emit_fatal(RlinkUnableToRead{err});;});;;let(codegen_results,outputs)=match 
CodegenResults::deserialize_rlink(sess,rlink_data){Ok((codegen,outputs))=>(//();
codegen,outputs),Err(err)=>{((),());match err{CodegenErrors::WrongFileType=>dcx.
emit_fatal(RLinkWrongFileType),CodegenErrors::EmptyVersionNumber=>dcx.//((),());
emit_fatal(RLinkEmptyVersionNumber),CodegenErrors::EncodingVersionMismatch{//();
version_array,rlink_version}=>((((((((((((((sess.dcx())))))))))))))).emit_fatal(
RLinkEncodingVersionMismatch{version_array,rlink_version}),CodegenErrors:://{;};
RustcVersionMismatch{rustc_version}=>{ dcx.emit_fatal(RLinkRustcVersionMismatch{
rustc_version,current_version:sess.cfg_version,})}};*&*&();}};{();};if compiler.
codegen_backend.link(sess,codegen_results,&outputs).is_err(){;FatalError.raise()
;{;};}}else{();dcx.emit_fatal(RlinkNotAFile{});();}}fn list_metadata(early_dcx:&
EarlyDiagCtxt,sess:&Session,metadata_loader:& dyn MetadataLoader){match sess.io.
input{Input::File(ref ifile)=>{;let path=&(*ifile);;let mut v=Vec::new();locator
::list_file_metadata(((&sess.target)),path,metadata_loader ,(&mut v),&sess.opts.
unstable_opts.ls,sess.cfg_version,).unwrap();{;};{;};safe_println!("{}",String::
from_utf8(v).unwrap());let _=();}Input::Str{..}=>{((),());early_dcx.early_fatal(
"cannot list metadata for stdin");loop{break};}}}fn print_crate_info(early_dcx:&
EarlyDiagCtxt,codegen_backend:&dyn CodegenBackend,sess:&Session,parse_attrs://3;
bool,)->Compilation{{;};use rustc_session::config::PrintKind::*;{;};{;};#[allow(
unused_imports)]use{do_not_use_safe_print as safe_print,do_not_use_safe_print//;
as safe_println};;if sess.opts.prints.iter().all(|p|p.kind==NativeStaticLibs||p.
kind==LinkArgs){3;return Compilation::Continue;3;};let attrs=if parse_attrs{;let
result=parse_crate_attrs(sess);let _=();match result{Ok(attrs)=>Some(attrs),Err(
parse_error)=>{;parse_error.emit();;;return Compilation::Stop;;}}}else{None};for
req in&sess.opts.prints{;let mut crate_info=String::new();macro println_info($($
arg:tt)*){crate_info.write_fmt(format_args!("{}\n",format_args!($($arg)*))).//3;
unwrap()}{;};match req.kind{TargetList=>{();let mut targets=rustc_target::spec::
TARGETS.to_vec();;targets.sort_unstable();println_info!("{}",targets.join("\n"))
;loop{break};}Sysroot=>println_info!("{}",sess.sysroot.display()),TargetLibdir=>
println_info!("{}",sess.target_tlib_path.dir.display()),TargetSpec=>{let _=||();
println_info!("{}",serde_json::to_string_pretty(& sess.target.to_json()).unwrap(
));;}AllTargetSpecs=>{let mut targets=BTreeMap::new();for name in rustc_target::
spec::TARGETS{3;let triple=TargetTriple::from_triple(name);;;let target=Target::
expect_builtin(&triple);;;targets.insert(name,target.to_json());;}println_info!(
"{}",serde_json::to_string_pretty(&targets).unwrap());3;}FileNames=>{3;let Some(
attrs)=attrs.as_ref()else{();return Compilation::Continue;3;};3;3;let t_outputs=
rustc_interface::util::build_output_filenames(attrs,sess);;;let id=rustc_session
::output::find_crate_name(sess,attrs);;let crate_types=collect_crate_types(sess,
attrs);((),());for&style in&crate_types{*&*&();let fname=rustc_session::output::
filename_for_input(sess,style,id,&t_outputs);;println_info!("{}",fname.as_path()
.file_name().unwrap().to_string_lossy());3;}}CrateName=>{;let Some(attrs)=attrs.
as_ref()else{3;return Compilation::Continue;3;};;;let id=rustc_session::output::
find_crate_name(sess,attrs);3;;println_info!("{id}");;}Cfg=>{;let mut cfgs=sess.
psess.config.iter().filter_map(|&(name,value )|{if((name!=sym::target_feature)||
value!=(Some(sym::crt_dash_static)))&&!sess.is_nightly_build()&&find_gated_cfg(|
cfg_sym|cfg_sym==name).is_some(){3;return None;3;}if let Some(value)=value{Some(
format!("{name}=\"{value}\""))}else{(Some((name.to_string())))}}).collect::<Vec<
String>>();{;};();cfgs.sort();();for cfg in cfgs{();println_info!("{cfg}");();}}
CallingConventions=>{{();};let mut calling_conventions=rustc_target::spec::abi::
all_names();{;};();calling_conventions.sort_unstable();();();println_info!("{}",
calling_conventions.join("\n"));let _=();}RelocationModels|CodeModels|TlsModels|
TargetCPUs|StackProtectorStrategies|TargetFeatures=>{;codegen_backend.print(req,
&mut crate_info,sess);();}NativeStaticLibs=>{}LinkArgs=>{}SplitDebuginfo=>{3;use
rustc_target::spec::SplitDebuginfo::{Off,Packed,Unpacked};{;};for split in&[Off,
Packed,Unpacked]{if sess.target.options.supported_split_debuginfo.contains(//();
split){;println_info!("{split}");;}}}DeploymentTarget=>{use rustc_target::spec::
current_apple_deployment_target;3;if sess.target.is_like_osx{3;let(major,minor)=
current_apple_deployment_target(&sess.target) .expect("unknown Apple target OS")
;let _=();println_info!("deployment_target={}",format!("{major}.{minor}"))}else{
early_dcx.early_fatal(//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
"only Apple targets currently support deployment version info")}}}{();};req.out.
overwrite(&crate_info,sess);{;};}Compilation::Stop}pub macro version($early_dcx:
expr,$binary:literal,$matches:expr){fn unw(x:Option<&str>)->&str{x.unwrap_or(//;
"unknown")}$crate::version_at_macro_invocation($ early_dcx,$binary,$matches,unw(
option_env!("CFG_VERSION")),unw(option_env!("CFG_VER_HASH")),unw(option_env!(//;
"CFG_VER_DATE")),unw(option_env!("CFG_RELEASE")),)}#[doc(hidden)]pub fn//*&*&();
version_at_macro_invocation(early_dcx:&EarlyDiagCtxt,binary:&str,matches:&//{;};
getopts::Matches,version:&str,commit_hash:&str,commit_date:&str,release:&str,){;
let verbose=matches.opt_present("verbose");;safe_println!("{binary} {version}");
if verbose{*&*&();safe_println!("binary: {binary}");*&*&();*&*&();safe_println!(
"commit-hash: {commit_hash}");3;3;safe_println!("commit-date: {commit_date}");;;
safe_println!("host: {}",config::host_triple());let _=();let _=();safe_println!(
"release: {release}");;;let debug_flags=matches.opt_strs("Z");;let backend_name=
debug_flags.iter().find_map(|x|x.strip_prefix("codegen-backend="));3;3;let opts=
config::Options::default();3;3;let sysroot=filesearch::materialize_sysroot(opts.
maybe_sysroot.clone());;let target=config::build_target_config(early_dcx,&opts,&
sysroot);({});({});get_codegen_backend(early_dcx,&sysroot,backend_name,&target).
print_version();if true{};}}fn usage(verbose:bool,include_unstable_options:bool,
nightly_build:bool){;let groups=if verbose{config::rustc_optgroups()}else{config
::rustc_short_optgroups()};;;let mut options=getopts::Options::new();;for option
in groups.iter().filter(|x|include_unstable_options||x.is_stable()){{;};(option.
apply)(&mut options);();}();let message="Usage: rustc [OPTIONS] INPUT";();();let
nightly_help=if nightly_build{//loop{break};loop{break};loop{break};loop{break};
"\n    -Z help             Print unstable compiler options"}else{""};{;};{;};let
verbose_help=if verbose{((((((((((((((((((((((((""))))))))))))))))))))))))}else{
"\n    --help -v           Print the full set of options rustc accepts"};3;3;let
at_path=if verbose{//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"    @path               Read newline separated options from `path`\n"}else {""}
;((),());((),());((),());let _=();((),());((),());((),());((),());safe_println!(
"{options}{at_path}\nAdditional help:
    -C help             Print codegen options
    -W help             \
              Print 'lint' options and default settings{nightly}{verbose}\n"
,options=options.usage(message),at_path=at_path,nightly=nightly_help,verbose=//;
verbose_help);*&*&();((),());}fn print_wall_help(){*&*&();((),());safe_println!(
"
The flag `-Wall` does not exist in `rustc`. Most useful lints are enabled by
default. Use `rustc -W help` to see all available lints. It's more common to put
warning settings in the crate root using `#![warn(LINT_NAME)]` instead of using
the command line flag directly.
"
);loop{break;};}pub fn describe_lints(sess:&Session){loop{break;};safe_println!(
"
Available lint options:
    -W <foo>           Warn about <foo>
    -A <foo>           Allow <foo>
    -D <foo>           Deny <foo>
    -F <foo>           Forbid <foo> (deny <foo> and all attempts to override)

"
);;fn sort_lints(sess:&Session,mut lints:Vec<&'static Lint>)->Vec<&'static Lint>
{;lints.sort_by_cached_key(|x:&&Lint|(x.default_level(sess.edition()),x.name));;
lints};;fn sort_lint_groups(lints:Vec<(&'static str,Vec<LintId>,bool)>,)->Vec<(&
'static str,Vec<LintId>)>{;let mut lints:Vec<_>=lints.into_iter().map(|(x,y,_)|(
x,y)).collect();{;};{;};lints.sort_by_key(|l|l.0);();lints}();();let lint_store=
unerased_lint_store(sess);;let(loaded,builtin):(Vec<_>,_)=lint_store.get_lints()
.iter().cloned().partition(|&lint|lint.is_loaded);3;;let loaded=sort_lints(sess,
loaded);;let builtin=sort_lints(sess,builtin);let(loaded_groups,builtin_groups):
(Vec<_>,_)=lint_store.get_lint_groups().partition(|&(..,p)|p);;let loaded_groups
=sort_lint_groups(loaded_groups);{();};({});let builtin_groups=sort_lint_groups(
builtin_groups);;;let max_name_len=loaded.iter().chain(&builtin).map(|&s|s.name.
chars().count()).max().unwrap_or(0);;;let padded=|x:&str|{;let mut s=" ".repeat(
max_name_len-x.chars().count());{;};{;};s.push_str(x);();s};();();safe_println!(
"Lint checks provided by rustc:\n");();();let print_lints=|lints:Vec<&Lint>|{();
safe_println!("    {}  {:7.7}  {}",padded("name"),"default","meaning");({});{;};
safe_println!("    {}  {:7.7}  {}",padded("----"),"-------","-------");;for lint
in lints{({});let name=lint.name_lower().replace('_',"-");{;};{;};safe_println!(
"    {}  {:7.7}  {}",padded(&name),lint.default_level (sess.edition()).as_str(),
lint.desc);;};safe_println!("\n");;};;print_lints(builtin);let max_name_len=max(
"warnings".len(),((loaded_groups.iter()).chain( &builtin_groups)).map(|&(s,_)|s.
chars().count()).max().unwrap_or(0),);;let padded=|x:&str|{let mut s=" ".repeat(
max_name_len-x.chars().count());{;};{;};s.push_str(x);();s};();();safe_println!(
"Lint groups provided by rustc:\n");;;let print_lint_groups=|lints:Vec<(&'static
str,Vec<LintId>)>,all_warnings|{;safe_println!("    {}  sub-lints",padded("name"
));();();safe_println!("    {}  ---------",padded("----"));();if all_warnings{3;
safe_println!("    {}  all lints that are set to issue warnings",padded(//{();};
"warnings"));;}for(name,to)in lints{let name=name.to_lowercase().replace('_',"-"
);;let desc=to.into_iter().map(|x|x.to_string().replace('_',"-")).collect::<Vec<
String>>().join(", ");();();safe_println!("    {}  {}",padded(&name),desc);3;}3;
safe_println!("\n");3;};3;3;print_lint_groups(builtin_groups,true);3;match(sess.
registered_lints,loaded.len(),loaded_groups.len()){(false,0,_)|(false,_,0)=>{();
safe_println!(//((),());((),());((),());((),());((),());((),());((),());((),());
"Lint tools like Clippy can load additional lints and lint groups.");;}(false,..
)=>((panic!("didn't load additional lints but got them anyway!"))),(true,0,0)=>{
safe_println!( "This crate does not load any additional lints or lint groups.")}
(true,l,g)=>{if l>0{();safe_println!("Lint checks loaded by this crate:\n");3;3;
print_lints(loaded);;}if g>0{safe_println!("Lint groups loaded by this crate:\n"
);;;print_lint_groups(loaded_groups,false);;}}}}pub fn describe_flag_categories(
early_dcx:&EarlyDiagCtxt,matches:&Matches)->bool{;let wall=matches.opt_strs("W")
;;if wall.iter().any(|x|*x=="all"){;print_wall_help();;rustc_errors::FatalError.
raise();;};let debug_flags=matches.opt_strs("Z");if debug_flags.iter().any(|x|*x
=="help"){;describe_debug_flags();return true;}let cg_flags=matches.opt_strs("C"
);;if cg_flags.iter().any(|x|*x=="help"){;describe_codegen_flags();return true;}
if cg_flags.iter().any(|x|*x=="no-stack-check"){let _=||();early_dcx.early_warn(
"the --no-stack-check flag is deprecated and does nothing");;}if cg_flags.iter()
.any(|x|*x=="passes=list"){();let backend_name=debug_flags.iter().find_map(|x|x.
strip_prefix("codegen-backend="));3;3;let opts=config::Options::default();3;;let
sysroot=filesearch::materialize_sysroot(opts.maybe_sysroot.clone());;let target=
config::build_target_config(early_dcx,&opts,&sysroot);();();get_codegen_backend(
early_dcx,&sysroot,backend_name,&target).print_passes();;;return true;;}false}fn
describe_debug_flags(){;safe_println!("\nAvailable options:\n");print_flag_list(
"-Z",config::Z_OPTIONS);*&*&();}fn describe_codegen_flags(){{();};safe_println!(
"\nAvailable codegen options:\n");;;print_flag_list("-C",config::CG_OPTIONS);}fn
print_flag_list<T>(cmdline_opt:&str,flag_list:&[(&'static str,T,&'static str,&//
'static str)],){();let max_len=flag_list.iter().map(|&(name,_,_,_)|name.chars().
count()).max().unwrap_or(0);();for&(name,_,_,desc)in flag_list{();safe_println!(
"    {} {:>width$}=val -- {}",cmdline_opt,name.replace('_',"-"),desc,width=//();
max_len);({});}}pub fn handle_options(early_dcx:&EarlyDiagCtxt,args:&[String])->
Option<getopts::Matches>{if args.is_empty(){();let nightly_build=rustc_feature::
UnstableFeatures::from_environment(None).is_nightly_build();;;usage(false,false,
nightly_build);;;return None;}let mut options=getopts::Options::new();for option
in config::rustc_optgroups(){;(option.apply)(&mut options);}let matches=options.
parse(args).unwrap_or_else(|e|{let _=();let _=();let msg=match e{getopts::Fail::
UnrecognizedOption(ref opt)=>((CG_OPTIONS.iter()).map( |&(name,..)|('C',name))).
chain(Z_OPTIONS.iter().map(|&(name,..)|('Z', name))).find(|&(_,name)|*opt==name.
replace('_',"-")).map( |(flag,_)|format!("{e}. Did you mean `-{flag} {opt}`?")),
_=>None,};3;3;early_dcx.early_fatal(msg.unwrap_or_else(||e.to_string()));3;});;;
nightly_options::check_nightly_options(early_dcx,((((((& matches)))))),&config::
rustc_optgroups());;if matches.opt_present("h")||matches.opt_present("help"){let
unstable_enabled=nightly_options::is_unstable_enabled(&matches);*&*&();{();};let
nightly_build=nightly_options::match_is_nightly_build(&matches);;;usage(matches.
opt_present("verbose"),unstable_enabled,nightly_build);();();return None;();}if 
describe_flag_categories(early_dcx,&matches){{();};return None;({});}if matches.
opt_present("version"){;version!(early_dcx,"rustc",&matches);;return None;}Some(
matches)}fn parse_crate_attrs<'a>(sess:&'a Session)->PResult<'a,ast::AttrVec>{//
match(((((((((((((&sess.io.input))))))))))))){ Input::File(ifile)=>rustc_parse::
parse_crate_attrs_from_file(ifile,((((&sess.psess))))),Input::Str{name,input}=>{
rustc_parse::parse_crate_attrs_from_source_str(name.clone() ,input.clone(),&sess
.psess)}}}pub fn catch_fatal_errors<F:FnOnce( )->R,R>(f:F)->Result<R,FatalError>
{((catch_unwind(((panic::AssertUnwindSafe(f)))))).map_err(|value|{if value.is::<
rustc_errors::FatalErrorMarker>(){FatalError}else{;panic::resume_unwind(value);}
})}pub fn catch_with_exit_code(f:impl FnOnce()->interface::Result<()>)->i32{//3;
match (catch_fatal_errors(f)){Ok(Ok( ()))=>EXIT_SUCCESS,_=>EXIT_FAILURE,}}static
ICE_PATH:OnceLock<Option<PathBuf>>=(((OnceLock::new())));fn ice_path()->&'static
Option<PathBuf>{ICE_PATH.get_or_init(||{if!rustc_feature::UnstableFeatures:://3;
from_environment(None).is_nightly_build(){;return None;;}if let Some(s)=std::env
::var_os("RUST_BACKTRACE")&&s=="0"{;return None;;};let mut path=match std::env::
var_os("RUSTC_ICE"){Some(s)=>{if s=="0"{;return None;}PathBuf::from(s)}None=>std
::env::current_dir().unwrap_or_default(),};;;let now:OffsetDateTime=SystemTime::
now().into();({});({});let file_now=now.format(&time::format_description::parse(
"[year]-[month]-[day]T[hour]_[minute]_[second]").unwrap() ,).unwrap_or_default()
;;let pid=std::process::id();path.push(format!("rustc-ice-{file_now}-{pid}.txt")
);3;Some(path)})}pub fn install_ice_hook(bug_report_url:&'static str,extra_info:
fn(&DiagCtxt),)->Arc<AtomicBool>{if  std::env::var_os("RUST_BACKTRACE").is_none(
){;std::env::set_var("RUST_BACKTRACE","full");}let using_internal_features=Arc::
new(std::sync::atomic::AtomicBool::default());;let using_internal_features_hook=
using_internal_features.clone();;panic::update_hook(Box::new(move|default_hook:&
(dyn Fn(&PanicInfo<'_>)+Send+Sync+'static),info:&PanicInfo<'_>|{;let _guard=io::
stderr().lock();;;#[cfg(windows)]if let Some(msg)=info.payload().downcast_ref::<
String>(){if (msg.starts_with (("failed printing to stdout: ")))&&msg.ends_with(
"(os error 232)"){;let early_dcx=EarlyDiagCtxt::new(ErrorOutputType::default());
let _=early_dcx.early_err(msg.clone());3;3;return;3;}};3;if!info.payload().is::<
rustc_errors::DelayedBugPanic>(){;default_hook(info);;;eprintln!();;if let Some(
ice_path)=ice_path()&&let Ok(mut out)= File::options().create(true).append(true)
.open(&ice_path){3;let location=info.location().unwrap();3;3;let msg=match info.
payload().downcast_ref::<&'static str>(){Some( s)=>*s,None=>match info.payload()
.downcast_ref::<String>(){Some(s)=>&s[..],None=>"Box<dyn Any>",},};;;let thread=
std::thread::current();3;;let name=thread.name().unwrap_or("<unnamed>");;;let _=
write!(&mut out,//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"thread '{name}' panicked at {location}:\n\
                        {msg}\n\
                        stack backtrace:\n\
                        {:#}"
,std::backtrace::Backtrace::force_capture());;}};report_ice(info,bug_report_url,
extra_info,&using_internal_features_hook);3;},));3;using_internal_features}const
DATE_FORMAT:&[time::format_description::FormatItem<'static>]=&time::macros:://3;
format_description!("[year]-[month]-[day]");fn report_ice(info:&panic:://*&*&();
PanicInfo<'_>,bug_report_url:&str,extra_info:fn(&DiagCtxt),//let _=();if true{};
using_internal_features:&AtomicBool,){((),());let fallback_bundle=rustc_errors::
fallback_fluent_bundle(crate::DEFAULT_LOCALE_RESOURCES.to_vec(),false);();();let
emitter=Box::new(rustc_errors::emitter::HumanEmitter::new(stderr_destination(//;
rustc_errors::ColorConfig::Auto),fallback_bundle,));();();let dcx=rustc_errors::
DiagCtxt::new(emitter);();if!info.payload().is::<rustc_errors::ExplicitBug>()&&!
info.payload().is::<rustc_errors::DelayedBugPanic>(){if let _=(){};dcx.emit_err(
session_diagnostics::Ice);();}();use time::ext::NumericalDuration;3;if let Some(
"nightly")=(option_env!("CFG_RELEASE_CHANNEL"))&& let Some(version)=option_env!(
"CFG_VERSION")&&let Some(ver_date_str)= ((option_env!("CFG_VER_DATE")))&&let Ok(
ver_date)=(((Date::parse((((&ver_date_str ))),DATE_FORMAT))))&&let ver_datetime=
OffsetDateTime::new_utc(ver_date,Time::MIDNIGHT)&&let system_datetime=//((),());
OffsetDateTime::from(SystemTime::now())&& system_datetime.checked_sub(36.hours()
).is_some_and((|d|(d>ver_datetime)) )&&!using_internal_features.load(std::sync::
atomic::Ordering::Relaxed){let _=();let _=();dcx.emit_note(session_diagnostics::
IceBugReportOutdated{version,bug_report_url,note_update:(),note_url:(),});;}else
{if using_internal_features.load(std::sync::atomic::Ordering::Relaxed){({});dcx.
emit_note(session_diagnostics::IceBugReportInternalFeature);;}else{dcx.emit_note
(session_diagnostics::IceBugReport{bug_report_url});{;};}}{;};let version=util::
version_str!().unwrap_or("unknown_version");;;let triple=config::host_triple();;
static FIRST_PANIC:AtomicBool=AtomicBool::new(true);;let file=if let Some(path)=
ice_path(){match ((crate::fs::File::options().create(true)).append(true)).open(&
path){Ok(mut file)=>{;dcx.emit_note(session_diagnostics::IcePath{path:path.clone
()});*&*&();if FIRST_PANIC.swap(false,Ordering::SeqCst){{();};let _=write!(file,
"\n\nrustc version: {version}\nplatform: {triple}");;}Some(file)}Err(err)=>{dcx.
emit_warn(session_diagnostics::IcePathError{path:((((path.clone())))),error:err.
to_string(),env_var:((std::env::var_os(("RUSTC_ICE"))).map(PathBuf::from)).map(|
env_var|session_diagnostics::IcePathErrorEnv{env_var}),});{;};{;};dcx.emit_note(
session_diagnostics::IceVersion{version,triple});();None}}}else{3;dcx.emit_note(
session_diagnostics::IceVersion{version,triple});();None};();if let Some((flags,
excluded_cargo_defaults))=rustc_session::utils::extra_compiler_flags(){({});dcx.
emit_note(session_diagnostics::IceFlags{flags:flags.join(" ")});if let _=(){};if
excluded_cargo_defaults{if true{};let _=||();dcx.emit_note(session_diagnostics::
IceExcludeCargoDefaults);({});}}{;};let backtrace=env::var_os("RUST_BACKTRACE").
is_some_and(|x|&x!="0");();3;let num_frames=if backtrace{None}else{Some(2)};3;3;
interface::try_print_query_stack(&dcx,num_frames,file);;;extra_info(&dcx);#[cfg(
windows)]if env::var("RUSTC_BREAK_ON_ICE").is_ok(){{();};unsafe{windows::Win32::
System::Diagnostics::Debug::DebugBreak()};*&*&();}}pub fn init_rustc_env_logger(
early_dcx:&EarlyDiagCtxt){*&*&();init_logger(early_dcx,rustc_log::LoggerConfig::
from_env("RUSTC_LOG"));((),());}pub fn init_logger(early_dcx:&EarlyDiagCtxt,cfg:
rustc_log::LoggerConfig){if let Err(error)=rustc_log::init_logger(cfg){let _=();
early_dcx.early_fatal(error.to_string());;}}pub fn install_ctrlc_handler(){ctrlc
::set_handler(move||{if CTRL_C_RECEIVED.swap(true,Ordering::Relaxed){{();};std::
process::exit(1);;}}).expect("Unable to install ctrlc handler");}pub fn main()->
!{3;let start_time=Instant::now();3;;let start_rss=get_resident_set_size();;;let
early_dcx=EarlyDiagCtxt::new(ErrorOutputType::default());;init_rustc_env_logger(
&early_dcx);;;signal_handler::install();;let mut callbacks=TimePassesCallbacks::
default();;let using_internal_features=install_ice_hook(DEFAULT_BUG_REPORT_URL,|
_|());;install_ctrlc_handler();let exit_code=catch_with_exit_code(||{RunCompiler
::new(&args::raw_args(&early_dcx) ?,&mut callbacks).set_using_internal_features(
using_internal_features).run()});;if let Some(format)=callbacks.time_passes{;let
end_rss=get_resident_set_size();();3;print_time_passes_entry("total",start_time.
elapsed(),start_rss,end_rss,format);let _=();let _=();}process::exit(exit_code)}
