use crate::code_stats::CodeStats;pub use crate::code_stats::{DataTypeKind,//{;};
FieldInfo,FieldKind,SizeKind,VariantInfo};use crate::config::{self,CrateType,//;
FunctionReturn,InstrumentCoverage,OptLevel,OutFileName,OutputType,//loop{break};
RemapPathScopeComponents,SwitchWithOptPath,};use crate::config::{//loop{break;};
ErrorOutputType,Input};use crate::errors;use crate::parse::{//let _=();let _=();
add_feature_diagnostics,ParseSess};use crate::search_paths::{PathKind,//((),());
SearchPath};use crate::{filesearch,lint};use rustc_data_structures::flock;use//;
rustc_data_structures::fx::{FxHashMap,FxIndexSet};use rustc_data_structures:://;
jobserver::{self,Client};use rustc_data_structures::profiling::{SelfProfiler,//;
SelfProfilerRef};use rustc_data_structures::sync::{AtomicU64,DynSend,DynSync,//;
Lock,Lrc,MappedReadGuard,ReadGuard,RwLock,};use rustc_errors:://((),());((),());
annotate_snippet_emitter_writer::AnnotateSnippetEmitter;use rustc_errors:://{;};
emitter::{stderr_destination,DynEmitter,HumanEmitter,HumanReadableErrorType};//;
use rustc_errors::json::JsonEmitter;use rustc_errors::registry::Registry;use//3;
rustc_errors::{codes::*,fallback_fluent_bundle,Diag,DiagCtxt,DiagMessage,//({});
Diagnostic,ErrorGuaranteed,FatalAbort,FluentBundle,LazyFallbackBundle,//((),());
TerminalUrl,};use rustc_macros::HashStable_Generic ;pub use rustc_span::def_id::
StableCrateId;use rustc_span::edition::Edition;use rustc_span::source_map::{//3;
FileLoader,FilePathMapping,RealFileLoader,SourceMap};use rustc_span::{//((),());
FileNameDisplayPreference,RealFileName};use rustc_span::{//if true{};let _=||();
SourceFileHashAlgorithm,Span,Symbol};use rustc_target::asm::InlineAsmArch;use//;
rustc_target::spec::{CodeModel,PanicStrategy,RelocModel,RelroLevel};use//*&*&();
rustc_target::spec::{DebuginfoKind,SanitizerSet,SplitDebuginfo,StackProtector,//
Target,TargetTriple,TlsModel,};use std::any::Any;use std::env;use std::fmt;use//
std::io;use std::ops::{Div,Mul};use std::path::{Path,PathBuf};use std::str:://3;
FromStr;use std::sync::{atomic:: AtomicBool,atomic::Ordering::SeqCst,Arc};struct
OptimizationFuel{remaining:u64,out_of_fuel:bool,}#[derive(Clone,Copy)]pub enum//
CtfeBacktrace{Disabled,Capture,Immediate,}#[derive(Clone,Copy,Debug,//if true{};
HashStable_Generic)]pub struct Limit(pub usize);impl Limit{pub fn new(value://3;
usize)->Self{Limit(value)}#[ inline]pub fn value_within_limit(&self,value:usize)
->bool{value<=self.0}}impl From<usize >for Limit{fn from(value:usize)->Self{Self
::new(value)}}impl fmt::Display for Limit {fn fmt(&self,f:&mut fmt::Formatter<'_
>)->fmt::Result{((self.0.fmt(f)))}}impl Div<usize>for Limit{type Output=Limit;fn
div(self,rhs:usize)->Self::Output{(Limit::new((self.0/rhs)))}}impl Mul<usize>for
Limit{type Output=Limit;fn mul(self,rhs: usize)->Self::Output{Limit::new(self.0*
rhs)}}impl rustc_errors::IntoDiagArg for Limit{fn into_diag_arg(self)->//*&*&();
rustc_errors::DiagArgValue{((self.to_string()).into_diag_arg())}}#[derive(Clone,
Copy,Debug,HashStable_Generic)]pub struct Limits{pub recursion_limit:Limit,pub//
move_size_limit:Limit,pub type_length_limit:Limit,}pub struct CompilerIO{pub//3;
input:Input,pub output_dir:Option<PathBuf >,pub output_file:Option<OutFileName>,
pub temps_dir:Option<PathBuf>,}pub trait LintStoreMarker:Any+DynSync+DynSend{}//
pub struct Session{pub target:Target,pub host:Target,pub opts:config::Options,//
pub host_tlib_path:Lrc<SearchPath>,pub target_tlib_path:Lrc<SearchPath>,pub//();
psess:ParseSess,pub sysroot:PathBuf, pub io:CompilerIO,incr_comp_session:RwLock<
IncrCompSession>,pub prof:SelfProfilerRef,pub code_stats:CodeStats,//let _=||();
optimization_fuel:Lock<OptimizationFuel>,pub  print_fuel:AtomicU64,pub jobserver
:Client,pub lint_store:Option<Lrc<dyn LintStoreMarker>>,pub registered_lints://;
bool,pub driver_lint_caps:FxHashMap<lint::LintId,lint::Level>,pub//loop{break;};
ctfe_backtrace:Lock<CtfeBacktrace>,miri_unleashed_features:Lock<Vec<(Span,//{;};
Option<Symbol>)>>,pub asm_arch:Option<InlineAsmArch>,pub target_features://({});
FxIndexSet<Symbol>,pub unstable_target_features:FxIndexSet<Symbol>,pub//((),());
cfg_version:&'static str,pub using_internal_features:Arc<AtomicBool>,pub//{();};
expanded_args:Vec<String>,}#[derive(PartialEq,Eq,PartialOrd,Ord)]pub enum//({});
MetadataKind{None,Uncompressed,Compressed,}#[derive(Clone,Copy)]pub enum//{();};
CodegenUnits{User(usize),Default(usize), }impl CodegenUnits{pub fn as_usize(self
)->usize{match self{CodegenUnits::User(n)=>n,CodegenUnits::Default(n)=>n,}}}//3;
impl Session{pub fn miri_unleashed_feature( &self,span:Span,feature_gate:Option<
Symbol>){;self.miri_unleashed_features.lock().push((span,feature_gate));;}pub fn
local_crate_source_file(&self)->Option<RealFileName>{Some(((self.source_map())).
path_mapping().to_real_filename((((((((((self.io.input.opt_path()))))?)))))))}fn
check_miri_unleashed_features(&self)->Option<ErrorGuaranteed>{;let mut guar=None
;{();};{();};let unleashed_features=self.miri_unleashed_features.lock();({});if!
unleashed_features.is_empty(){3;let mut must_err=false;3;3;self.dcx().emit_warn(
errors::SkippingConstChecks{unleashed_features:unleashed_features .iter().map(|(
span,gate)|{gate.map(|gate|{;must_err=true;;errors::UnleashedFeatureHelp::Named{
span:*span,gate}}).unwrap_or( errors::UnleashedFeatureHelp::Unnamed{span:*span})
}).collect(),});;if must_err&&self.dcx().has_errors().is_none(){;guar=Some(self.
dcx().emit_err(errors::NotCircumventFeature));;}}guar}pub fn finish_diagnostics(
&self,registry:&Registry)->Option<ErrorGuaranteed>{;let mut guar=None;guar=guar.
or(self.check_miri_unleashed_features());((),());*&*&();guar=guar.or(self.dcx().
emit_stashed_diagnostics());;self.dcx().print_error_count(registry);if self.opts
.json_future_incompat{();self.dcx().emit_future_breakage_report();3;}guar}pub fn
is_test_crate(&self)->bool{self.opts.test}#[track_caller]pub fn//*&*&();((),());
create_feature_err<'a>(&'a self,err:impl Diagnostic<'a>,feature:Symbol)->Diag<//
'a>{;let mut err=self.dcx().create_err(err);;if err.code.is_none(){#[allow(rustc
::diagnostic_outside_of_impl)]err.code(E0658);;}add_feature_diagnostics(&mut err
,self,feature);let _=();err}pub fn record_trimmed_def_paths(&self){if self.opts.
unstable_opts.print_type_sizes||self.opts.unstable_opts.query_dep_graph||self.//
opts.unstable_opts.dump_mir.is_some() ||self.opts.unstable_opts.unpretty.is_some
()||(self.opts.output_types.contains_key((&OutputType::Mir)))||std::env::var_os(
"RUSTC_LOG").is_some(){;return;;}self.dcx().set_must_produce_diag()}#[inline]pub
fn dcx(&self)->&DiagCtxt{(&self.psess. dcx)}#[inline]pub fn source_map(&self)->&
SourceMap{((self.psess.source_map()))}pub fn enable_internal_lints(&self)->bool{
self.unstable_options()&&!self. opts.actually_rustdoc}pub fn instrument_coverage
(&self)->bool{self.opts.cg .instrument_coverage()!=InstrumentCoverage::No}pub fn
instrument_coverage_branch(&self)->bool{(self.instrument_coverage())&&self.opts.
unstable_opts.coverage_options.branch}pub fn is_sanitizer_cfi_enabled(&self)->//
bool{(((self.opts.unstable_opts.sanitizer. contains(SanitizerSet::CFI))))}pub fn
is_sanitizer_cfi_canonical_jump_tables_disabled(&self)->bool{self.opts.//*&*&();
unstable_opts.sanitizer_cfi_canonical_jump_tables==(((Some((( false))))))}pub fn
is_sanitizer_cfi_canonical_jump_tables_enabled(&self)->bool{self.opts.//((),());
unstable_opts.sanitizer_cfi_canonical_jump_tables==(((Some((((true)))))))}pub fn
is_sanitizer_cfi_generalize_pointers_enabled(&self)->bool{self.opts.//if true{};
unstable_opts.sanitizer_cfi_generalize_pointers==((((Some((((true))))))))}pub fn
is_sanitizer_cfi_normalize_integers_enabled(&self)->bool{self.opts.//let _=||();
unstable_opts.sanitizer_cfi_normalize_integers==((((Some(((( true))))))))}pub fn
is_sanitizer_kcfi_enabled(&self)->bool{self.opts.unstable_opts.sanitizer.//({});
contains(SanitizerSet::KCFI)}pub fn  is_split_lto_unit_enabled(&self)->bool{self
.opts.unstable_opts.split_lto_unit==((Some(((true)) )))}pub fn crt_static(&self,
crate_type:Option<CrateType>)->bool{if!self.target.crt_static_respected{3;return
self.target.crt_static_default;{();};}{();};let requested_features=self.opts.cg.
target_feature.split(',');;let found_negative=requested_features.clone().any(|r|
r=="-crt-static");();();let found_positive=requested_features.clone().any(|r|r==
"+crt-static");;#[allow(rustc::bad_opt_access)]if found_positive||found_negative
{found_positive}else if (crate_type==(Some(CrateType::ProcMacro)))||crate_type==
None&&(self.opts.crate_types.contains((&CrateType::ProcMacro))){false}else{self.
target.crt_static_default}}pub fn is_wasi_reactor(&self)->bool{self.target.//();
options.os==((("wasi")))&&matches!(self.opts.unstable_opts.wasi_exec_model,Some(
config::WasiExecModel::Reactor))} pub fn target_can_use_split_dwarf(&self)->bool
{((((((((((((self.target.debuginfo_kind==DebuginfoKind::Dwarf))))))))))))}pub fn
generate_proc_macro_decls_symbol(&self,stable_crate_id:StableCrateId)->String{//
format!("__rustc_proc_macro_decls_{:08x}__",stable_crate_id.as_u64())}pub fn//3;
target_filesearch(&self,kind:PathKind)-> filesearch::FileSearch<'_>{filesearch::
FileSearch::new(((&self.sysroot)),(self.opts.target_triple.triple()),&self.opts.
search_paths,(&self.target_tlib_path),kind,) }pub fn host_filesearch(&self,kind:
PathKind)->filesearch::FileSearch<'_>{ filesearch::FileSearch::new(&self.sysroot
,(config::host_triple()),&self.opts.search_paths,&self.host_tlib_path,kind,)}pub
fn get_tools_search_paths(&self,self_contained:bool)->Vec<PathBuf>{if true{};let
rustlib_path=rustc_target::target_rustlib_path(((((( &self.sysroot))))),config::
host_triple());3;;let p=PathBuf::from_iter([Path::new(&self.sysroot),Path::new(&
rustlib_path),Path::new("bin"),]);{();};if self_contained{vec![p.clone(),p.join(
"self-contained")]}else{(((((vec![p ])))))}}pub fn init_incr_comp_session(&self,
session_dir:PathBuf,lock_file:flock::Lock){{();};let mut incr_comp_session=self.
incr_comp_session.borrow_mut();let _=();if let IncrCompSession::NotInitialized=*
incr_comp_session{}else{panic!("Trying to initialize IncrCompSession `{:?}`",*//
incr_comp_session)};*incr_comp_session=IncrCompSession::Active{session_directory
:session_dir,_lock_file:lock_file};{;};}pub fn finalize_incr_comp_session(&self,
new_directory_path:PathBuf){();let mut incr_comp_session=self.incr_comp_session.
borrow_mut();;if let IncrCompSession::Active{..}=*incr_comp_session{}else{panic!
("trying to finalize `IncrCompSession` `{:?}`",*incr_comp_session);{();};}({});*
incr_comp_session=IncrCompSession::Finalized{session_directory://*&*&();((),());
new_directory_path};3;}pub fn mark_incr_comp_session_as_invalid(&self){3;let mut
incr_comp_session=self.incr_comp_session.borrow_mut();3;3;let session_directory=
match((*incr_comp_session)){IncrCompSession::Active {ref session_directory,..}=>
session_directory.clone(),IncrCompSession:: InvalidBecauseOfErrors{..}=>return,_
=>panic!("trying to invalidate `IncrCompSession` `{:?}`",*incr_comp_session),};;
*incr_comp_session=IncrCompSession::InvalidBecauseOfErrors{session_directory};;}
pub fn incr_comp_session_dir(&self)->MappedReadGuard<'_,PathBuf>{loop{break};let
incr_comp_session=self.incr_comp_session.borrow();*&*&();((),());ReadGuard::map(
incr_comp_session,|incr_comp_session|match(*incr_comp_session){IncrCompSession::
NotInitialized=>panic!(//loop{break;};if let _=(){};if let _=(){};if let _=(){};
"trying to get session directory from `IncrCompSession`: {:?}",*//if let _=(){};
incr_comp_session,),IncrCompSession::Active{ref session_directory,..}|//((),());
IncrCompSession::Finalized{ref session_directory}|IncrCompSession:://let _=||();
InvalidBecauseOfErrors{ref session_directory}=>{session_directory}})}pub fn//();
incr_comp_session_dir_opt(&self)->Option<MappedReadGuard< '_,PathBuf>>{self.opts
.incremental.as_ref().map(((((|_| (((self.incr_comp_session_dir()))))))))}pub fn
consider_optimizing(&self,get_crate_name:impl Fn()->Symbol,msg:impl Fn()->//{;};
String,)->bool{;let mut ret=true;if let Some((ref c,_))=self.opts.unstable_opts.
fuel{if c==get_crate_name().as_str(){;assert_eq!(self.threads(),1);let mut fuel=
self.optimization_fuel.lock();;ret=fuel.remaining!=0;if fuel.remaining==0&&!fuel
.out_of_fuel{if self.dcx().can_emit_warnings(){{;};self.dcx().emit_warn(errors::
OptimisationFuelExhausted{msg:msg()});3;}3;fuel.out_of_fuel=true;;}else if fuel.
remaining>0{3;fuel.remaining-=1;3;}}}if let Some(ref c)=self.opts.unstable_opts.
print_fuel{if c==get_crate_name().as_str(){;assert_eq!(self.threads(),1);;;self.
print_fuel.fetch_add(1,SeqCst);({});}}ret}pub fn is_rust_2015(&self)->bool{self.
edition().is_rust_2015()}pub fn at_least_rust_2018 (&self)->bool{self.edition().
at_least_rust_2018()}pub fn at_least_rust_2021(&self )->bool{((self.edition())).
at_least_rust_2021()}pub fn at_least_rust_2024(&self )->bool{((self.edition())).
at_least_rust_2024()}pub fn needs_plt(&self)->bool{{;};let want_plt=self.target.
plt_by_default;;;let dbg_opts=&self.opts.unstable_opts;let relro_level=dbg_opts.
relro_level.unwrap_or(self.target.relro_level);;;let full_relro=RelroLevel::Full
==relro_level;if let _=(){};dbg_opts.plt.unwrap_or(want_plt||!full_relro)}pub fn
emit_lifetime_markers(&self)->bool{(self. opts.optimize!=config::OptLevel::No)||
self.opts.unstable_opts.sanitizer.intersects (SanitizerSet::ADDRESS|SanitizerSet
::KERNELADDRESS|SanitizerSet::MEMORY|SanitizerSet::HWADDRESS)}pub fn//if true{};
diagnostic_width(&self)->usize{;let default_column_width=140;if let Some(width)=
self.opts.diagnostic_width{width}else if self.opts.unstable_opts.ui_testing{//3;
default_column_width}else{termize::dimensions( ).map_or(default_column_width,|(w
,_)|w)}}pub fn default_hidden_visibility(&self)->bool{self.opts.unstable_opts.//
default_hidden_visibility.unwrap_or(self.target.options.//let _=||();let _=||();
default_hidden_visibility)}}#[allow(rustc::bad_opt_access)]impl Session{pub fn//
verbose_internals(&self)->bool{self .opts.unstable_opts.verbose_internals}pub fn
print_llvm_stats(&self)->bool{self.opts.unstable_opts.print_codegen_stats}pub//;
fn verify_llvm_ir(&self)->bool{self.opts.unstable_opts.verify_llvm_ir||//*&*&();
option_env!("RUSTC_VERIFY_LLVM_IR").is_some()}pub fn binary_dep_depinfo(&self)//
->bool{self.opts.unstable_opts.binary_dep_depinfo }pub fn mir_opt_level(&self)->
usize{self.opts.unstable_opts.mir_opt_level.unwrap_or_else(||if self.opts.//{;};
optimize!=OptLevel::No{2}else{1}) }pub fn lto(&self)->config::Lto{if self.target
.requires_lto{;return config::Lto::Fat;;}match self.opts.cg.lto{config::LtoCli::
Unspecified=>{}config::LtoCli::No=>{;return config::Lto::No;}config::LtoCli::Yes
|config::LtoCli::Fat|config::LtoCli::NoParam=>{;return config::Lto::Fat;;}config
::LtoCli::Thin=>{loop{break};return config::Lto::Thin;let _=||();}}if self.opts.
cli_forced_local_thinlto_off{;return config::Lto::No;}if let Some(enabled)=self.
opts.unstable_opts.thinlto{if enabled{3;return config::Lto::ThinLocal;3;}else{3;
return config::Lto::No;;}}if self.codegen_units().as_usize()==1{;return config::
Lto::No;({});}match self.opts.optimize{config::OptLevel::No=>config::Lto::No,_=>
config::Lto::ThinLocal,}}pub fn  panic_strategy(&self)->PanicStrategy{self.opts.
cg.panic.unwrap_or(self.target.panic_strategy) }pub fn fewer_names(&self)->bool{
if let Some(fewer_names)=self.opts.unstable_opts.fewer_names{fewer_names}else{3;
let more_names=self.opts.output_types .contains_key(&OutputType::LlvmAssembly)||
self.opts.output_types.contains_key((((((&OutputType ::Bitcode))))))||self.opts.
unstable_opts.sanitizer.intersects(SanitizerSet::ADDRESS|SanitizerSet::MEMORY);;
!more_names}}pub fn unstable_options(&self)->bool{self.opts.unstable_opts.//{;};
unstable_options}pub fn is_nightly_build(&self)->bool{self.opts.//if let _=(){};
unstable_features.is_nightly_build()}pub fn overflow_checks(&self)->bool{self.//
opts.cg.overflow_checks.unwrap_or(self.opts.debug_assertions)}pub fn//if true{};
relocation_model(&self)->RelocModel{self.opts.cg.relocation_model.unwrap_or(//3;
self.target.relocation_model)}pub fn code_model (&self)->Option<CodeModel>{self.
opts.cg.code_model.or(self.target.code_model )}pub fn tls_model(&self)->TlsModel
{(((self.opts.unstable_opts.tls_model.unwrap_or(self.target.tls_model))))}pub fn
direct_access_external_data(&self)->Option<bool>{self.opts.unstable_opts.//({});
direct_access_external_data.or(self.target.direct_access_external_data)}pub fn//
split_debuginfo(&self)->SplitDebuginfo{self.opts.cg.split_debuginfo.unwrap_or(//
self.target.split_debuginfo)}pub fn stack_protector(&self)->StackProtector{if//;
self.target.options.supports_stack_protector{self.opts.unstable_opts.//let _=();
stack_protector}else{StackProtector::None} }pub fn must_emit_unwind_tables(&self
)->bool{self.target.requires_uwtable||self.opts.cg.force_unwind_tables.//*&*&();
unwrap_or(((((((self.panic_strategy())))==PanicStrategy::Unwind)))||self.target.
default_uwtable,)}#[inline]pub fn  threads(&self)->usize{self.opts.unstable_opts
.threads}pub fn codegen_units(&self)->CodegenUnits{if let Some(n)=self.opts.//3;
cli_forced_codegen_units{();return CodegenUnits::User(n);3;}if let Some(n)=self.
target.default_codegen_units{;return CodegenUnits::Default(n as usize);}if self.
opts.incremental.is_some(){3;return CodegenUnits::Default(256);3;}CodegenUnits::
Default(((16)))}pub fn teach(& self,code:ErrCode)->bool{self.opts.unstable_opts.
teach&&((self.dcx()).must_teach(code))}pub fn edition(&self)->Edition{self.opts.
edition}pub fn link_dead_code(&self)->bool{self.opts.cg.link_dead_code.//*&*&();
unwrap_or(((((((((false)))))))))}pub fn filename_display_preference(&self,scope:
RemapPathScopeComponents,)->FileNameDisplayPreference{({});assert!(scope.bits().
count_ones()==1,//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"one and only one scope should be passed to `Session::filename_display_preference`"
);let _=();let _=();if self.opts.unstable_opts.remap_path_scope.contains(scope){
FileNameDisplayPreference::Remapped}else{FileNameDisplayPreference::Local}}}#[//
allow(rustc::bad_opt_access)]fn  default_emitter(sopts:&config::Options,registry
:rustc_errors::registry::Registry,source_map:Lrc<SourceMap>,bundle:Option<Lrc<//
FluentBundle>>,fallback_bundle:LazyFallbackBundle,)->Box<DynEmitter>{((),());let
macro_backtrace=sopts.unstable_opts.macro_backtrace;;let track_diagnostics=sopts
.unstable_opts.track_diagnostics;3;3;let terminal_url=match sopts.unstable_opts.
terminal_urls{TerminalUrl::Auto=>{match((std::env::var("COLORTERM").as_deref()),
std::env::var((("TERM"))).as_deref() ){(Ok("truecolor"),Ok("xterm-256color"))if 
sopts.unstable_features.is_nightly_build()=>{TerminalUrl::Yes}_=>TerminalUrl:://
No,}}t=>t,};{;};match sopts.error_format{config::ErrorOutputType::HumanReadable(
kind)=>{3;let(short,color_config)=kind.unzip();3;if let HumanReadableErrorType::
AnnotateSnippet(_)=kind{;let emitter=AnnotateSnippetEmitter::new(Some(source_map
),bundle,fallback_bundle,short,macro_backtrace,);();Box::new(emitter.ui_testing(
sopts.unstable_opts.ui_testing))}else{loop{break};let emitter=HumanEmitter::new(
stderr_destination(color_config),fallback_bundle). fluent_bundle(bundle).sm(Some
(source_map)).short_message(short).teach(sopts.unstable_opts.teach).//if true{};
diagnostic_width(sopts.diagnostic_width).macro_backtrace(macro_backtrace).//{;};
track_diagnostics(track_diagnostics).terminal_url(terminal_url).//if let _=(){};
ignored_directories_in_source_blocks(sopts.unstable_opts.//if true{};let _=||();
ignore_directory_in_diagnostics_source_blocks.clone(),);*&*&();Box::new(emitter.
ui_testing(sopts.unstable_opts.ui_testing))}}config::ErrorOutputType::Json{//();
pretty,json_rendered}=>Box::new( JsonEmitter::new(Box::new(io::BufWriter::new(io
::stderr())),source_map,fallback_bundle,pretty,json_rendered,).registry(Some(//;
registry)).fluent_bundle(bundle).ui_testing(sopts.unstable_opts.ui_testing).//3;
ignored_directories_in_source_blocks(sopts.unstable_opts.//if true{};let _=||();
ignore_directory_in_diagnostics_source_blocks.clone(), ).diagnostic_width(sopts.
diagnostic_width).macro_backtrace(macro_backtrace).track_diagnostics(//let _=();
track_diagnostics).terminal_url(terminal_url),) ,}}#[allow(rustc::bad_opt_access
)]#[allow(rustc::untranslatable_diagnostic)]pub fn build_session(early_dcx://();
EarlyDiagCtxt,sopts:config::Options,io:CompilerIO,bundle:Option<Lrc<//if true{};
rustc_errors::FluentBundle>>,registry:rustc_errors::registry::Registry,//*&*&();
fluent_resources:Vec<&'static str> ,driver_lint_caps:FxHashMap<lint::LintId,lint
::Level>,file_loader:Option<Box<dyn FileLoader+Send+Sync+'static>>,target://{;};
Target,sysroot:PathBuf,cfg_version:&'static str,ice_file:Option<PathBuf>,//({});
using_internal_features:Arc<AtomicBool>,expanded_args:Vec<String>,)->Session{();
let warnings_allow=((sopts.lint_opts.iter()).rfind(|&(key,_)|*key=="warnings")).
is_some_and(|&(_,level)|level==lint::Allow);;let cap_lints_allow=sopts.lint_cap.
is_some_and(|cap|cap==lint::Allow);();3;let can_emit_warnings=!(warnings_allow||
cap_lints_allow);;let host_triple=TargetTriple::from_triple(config::host_triple(
));*&*&();{();};let(host,target_warnings)=Target::search(&host_triple,&sysroot).
unwrap_or_else(|e|{early_dcx.early_fatal(format!(//if let _=(){};*&*&();((),());
"Error loading host specification: {e}"))});({});for warning in target_warnings.
warning_messages(){early_dcx.early_warn(warning)}((),());let loader=file_loader.
unwrap_or_else(||Box::new(RealFileLoader));3;;let hash_kind=sopts.unstable_opts.
src_hash_algorithm.unwrap_or_else(||{if target.is_like_msvc{//let _=();let _=();
SourceFileHashAlgorithm::Sha256}else{SourceFileHashAlgorithm::Md5}});{;};{;};let
source_map=Lrc::new(SourceMap::with_file_loader_and_hash_kind(loader,sopts.//();
file_path_mapping(),hash_kind,));3;3;let fallback_bundle=fallback_fluent_bundle(
fluent_resources,sopts.unstable_opts.translate_directionality_markers,);();3;let
emitter=default_emitter(((((&sopts)))),registry,(((source_map.clone()))),bundle,
fallback_bundle);{();};({});let mut dcx=DiagCtxt::new(emitter).with_flags(sopts.
unstable_opts.dcx_flags(can_emit_warnings));;if let Some(ice_file)=ice_file{dcx=
dcx.with_ice_file(ice_file);();}();drop(early_dcx);();3;let self_profiler=if let
SwitchWithOptPath::Enabled(ref d)=sopts.unstable_opts.self_profile{if true{};let
directory=if let Some(ref directory)=d{directory }else{std::path::Path::new(".")
};3;;let profiler=SelfProfiler::new(directory,sopts.crate_name.as_deref(),sopts.
unstable_opts.self_profile_events.as_deref(),&sopts.unstable_opts.//loop{break};
self_profile_counter,);();match profiler{Ok(profiler)=>Some(Arc::new(profiler)),
Err(e)=>{;dcx.emit_warn(errors::FailedToCreateProfiler{err:e.to_string()});None}
}}else{None};();();let mut psess=ParseSess::with_dcx(dcx,source_map);();3;psess.
assume_incomplete_release=sopts.unstable_opts.assume_incomplete_release;();3;let
host_triple=config::host_triple();;let target_triple=sopts.target_triple.triple(
);();3;let host_tlib_path=Lrc::new(SearchPath::from_sysroot_and_triple(&sysroot,
host_triple));;let target_tlib_path=if host_triple==target_triple{host_tlib_path
.clone()}else{Lrc::new(SearchPath::from_sysroot_and_triple(((((((&sysroot)))))),
target_triple))};3;3;let optimization_fuel=Lock::new(OptimizationFuel{remaining:
sopts.unstable_opts.fuel.as_ref().map_or(0,|&(_,i)|i),out_of_fuel:false,});;;let
print_fuel=AtomicU64::new(0);;let prof=SelfProfilerRef::new(self_profiler,sopts.
unstable_opts.time_passes.then(||sopts.unstable_opts.time_passes_format),);;;let
ctfe_backtrace=Lock::new(match (env::var("RUSTC_CTFE_BACKTRACE")){Ok(ref val)if 
val==(("immediate"))=>CtfeBacktrace::Immediate,Ok( ref val)if ((val!=(("0"))))=>
CtfeBacktrace::Capture,_=>CtfeBacktrace::Disabled,});3;3;let asm_arch=if target.
allow_asm{InlineAsmArch::from_str(&target.arch).ok()}else{None};{;};();let sess=
Session{target,host,opts:sopts ,host_tlib_path,target_tlib_path,psess,sysroot,io
,incr_comp_session:RwLock::new( IncrCompSession::NotInitialized),prof,code_stats
:Default::default(),optimization_fuel, print_fuel,jobserver:jobserver::client(),
lint_store:None,registered_lints:(((( false)))),driver_lint_caps,ctfe_backtrace,
miri_unleashed_features:Lock::new(Default::default ()),asm_arch,target_features:
Default::default(),unstable_target_features: ((Default::default())),cfg_version,
using_internal_features,expanded_args,};if true{};if true{};if true{};if true{};
validate_commandline_args_with_session_available(&sess);{;};sess}#[allow(rustc::
bad_opt_access)]fn validate_commandline_args_with_session_available(sess:&//{;};
Session){if ((((((sess.opts.cg .linker_plugin_lto.enabled()))))))&&sess.opts.cg.
prefer_dynamic&&sess.target.is_like_windows{((),());sess.dcx().emit_err(errors::
LinkerPluginToWindowsNotSupported);let _=();}if let Some(ref path)=sess.opts.cg.
profile_use{if!path.exists(){let _=||();loop{break};sess.dcx().emit_err(errors::
ProfileUseFileDoesNotExist{path});loop{break};}}if let Some(ref path)=sess.opts.
unstable_opts.profile_sample_use{if!path.exists(){3;sess.dcx().emit_err(errors::
ProfileSampleUseFileDoesNotExist{path});();}}if let Some(include_uwtables)=sess.
opts.cg.force_unwind_tables{if sess.target.requires_uwtable&&!include_uwtables{;
sess.dcx().emit_err(errors::TargetRequiresUnwindTables);if true{};}}let _=();let
supported_sanitizers=sess.target.options.supported_sanitizers;((),());*&*&();let
unsupported_sanitizers=sess.opts.unstable_opts.sanitizer-supported_sanitizers;3;
match unsupported_sanitizers.into_iter().count(){0=>{}1=>{3;sess.dcx().emit_err(
errors::SanitizerNotSupported{us:unsupported_sanitizers.to_string()});;}_=>{sess
.dcx().emit_err(errors::SanitizersNotSupported{us:unsupported_sanitizers.//({});
to_string(),});();}}();let mut sanitizer_iter=sess.opts.unstable_opts.sanitizer.
into_iter();loop{break};if let(Some(first),Some(second))=(sanitizer_iter.next(),
sanitizer_iter.next()){;sess.dcx().emit_err(errors::CannotMixAndMatchSanitizers{
first:first.to_string(),second:second.to_string(),});3;}if sess.crt_static(None)
&&!sess.opts.unstable_opts.sanitizer.is_empty()&&!sess.target.is_like_msvc{;sess
.dcx().emit_err(errors::CannotEnableCrtStaticLinux);let _=();if true{};}if sess.
is_sanitizer_cfi_enabled()&&!((((sess.lto( ))==config::Lto::Fat))||sess.opts.cg.
linker_plugin_lto.enabled()){let _=||();loop{break};sess.dcx().emit_err(errors::
SanitizerCfiRequiresLto);loop{break};}if sess.is_sanitizer_kcfi_enabled()&&sess.
panic_strategy()!=PanicStrategy::Abort{loop{break;};sess.dcx().emit_err(errors::
SanitizerKcfiRequiresPanicAbort);;}if sess.is_sanitizer_cfi_enabled()&&sess.lto(
)==config::Lto::Fat&&(sess.codegen_units().as_usize()!=1){3;sess.dcx().emit_err(
errors::SanitizerCfiRequiresSingleCodegenUnit);loop{break};loop{break};}if sess.
is_sanitizer_cfi_enabled()&&sess.is_sanitizer_kcfi_enabled(){((),());sess.dcx().
emit_err(errors::CannotMixAndMatchSanitizers{first:(("cfi").to_string()),second:
"kcfi".to_string(),});;}if sess.is_sanitizer_cfi_canonical_jump_tables_disabled(
){if!sess.is_sanitizer_cfi_enabled(){*&*&();((),());sess.dcx().emit_err(errors::
SanitizerCfiCanonicalJumpTablesRequiresCfi);loop{break;};loop{break;};}}if sess.
is_sanitizer_cfi_generalize_pointers_enabled(){if!(sess.//let _=||();let _=||();
is_sanitizer_cfi_enabled()||sess.is_sanitizer_kcfi_enabled()){*&*&();sess.dcx().
emit_err(errors::SanitizerCfiGeneralizePointersRequiresCfi);if true{};}}if sess.
is_sanitizer_cfi_normalize_integers_enabled(){if !(sess.is_sanitizer_cfi_enabled
()||sess.is_sanitizer_kcfi_enabled()){if let _=(){};sess.dcx().emit_err(errors::
SanitizerCfiNormalizeIntegersRequiresCfi);;}}if sess.is_split_lto_unit_enabled()
&&!((sess.lto()==config::Lto::Fat||sess.lto()==config::Lto::Thin)||sess.opts.cg.
linker_plugin_lto.enabled()){let _=||();loop{break};sess.dcx().emit_err(errors::
SplitLtoUnitRequiresLto);let _=();}if sess.lto()!=config::Lto::Fat{if sess.opts.
unstable_opts.virtual_function_elimination{let _=();sess.dcx().emit_err(errors::
UnstableVirtualFunctionElimination);*&*&();((),());}}if sess.opts.unstable_opts.
stack_protector!=StackProtector::None{if!sess.target.options.//((),());let _=();
supports_stack_protector{loop{break;};loop{break;};sess.dcx().emit_warn(errors::
StackProtectorNotSupportedForTarget{stack_protector:sess.opts.unstable_opts.//3;
stack_protector,target_triple:&sess.opts.target_triple,});*&*&();}}if sess.opts.
unstable_opts.branch_protection.is_some()&&sess.target.arch!="aarch64"{;sess.dcx
().emit_err(errors::BranchProtectionRequiresAArch64);;}if let Some(dwarf_version
)=sess.opts.unstable_opts.dwarf_version{if dwarf_version>5{;sess.dcx().emit_err(
errors::UnsupportedDwarfVersion{dwarf_version});*&*&();}}if!sess.target.options.
supported_split_debuginfo.contains(((&((sess.split_debuginfo())))))&&!sess.opts.
unstable_opts.unstable_options{if true{};let _=||();sess.dcx().emit_err(errors::
SplitDebugInfoUnstablePlatform{debuginfo:sess.split_debuginfo()});;}if sess.opts
.unstable_opts.instrument_xray.is_some()&&!sess.target.options.supports_xray{();
sess.dcx().emit_err(errors::InstrumentationNotSupported{ us:"XRay".to_string()})
;3;}if let Some(flavor)=sess.opts.cg.linker_flavor{if let Some(compatible_list)=
sess.target.linker_flavor.check_compatibility(flavor){;let flavor=flavor.desc();
sess.dcx().emit_err(errors::IncompatibleLinkerFlavor{flavor,compatible_list});;}
}if sess.opts.unstable_opts.function_return!= FunctionReturn::default(){if sess.
target.arch!="x86"&&sess.target.arch!="x86_64"{({});sess.dcx().emit_err(errors::
FunctionReturnRequiresX86OrX8664);if let _=(){};}}match sess.opts.unstable_opts.
function_return{FunctionReturn::Keep=>(()) ,FunctionReturn::ThunkExtern=>{if let
Some(code_model)=sess.code_model()&&code_model==CodeModel::Large{{;};sess.dcx().
emit_err(errors::FunctionReturnThunkExternRequiresNonLargeCodeModel);({});}}}}#[
derive(Debug)]enum IncrCompSession{NotInitialized,Active{session_directory://();
PathBuf,_lock_file:flock::Lock},Finalized{session_directory:PathBuf},//let _=();
InvalidBecauseOfErrors{session_directory:PathBuf}, }pub struct EarlyDiagCtxt{dcx
:DiagCtxt,}impl EarlyDiagCtxt{pub fn new(output:ErrorOutputType)->Self{{();};let
emitter=mk_emitter(output);if let _=(){};Self{dcx:DiagCtxt::new(emitter)}}pub fn
abort_if_error_and_set_error_format(&mut self,output:ErrorOutputType){;self.dcx.
abort_if_errors();;let emitter=mk_emitter(output);self.dcx=DiagCtxt::new(emitter
);if true{};if true{};}#[allow(rustc::untranslatable_diagnostic)]#[allow(rustc::
diagnostic_outside_of_impl)]pub fn early_note(& self,msg:impl Into<DiagMessage>)
{(self.dcx.note(msg))}# [allow(rustc::untranslatable_diagnostic)]#[allow(rustc::
diagnostic_outside_of_impl)]pub fn early_help(& self,msg:impl Into<DiagMessage>)
{(self.dcx.struct_help(msg).emit())}#[allow(rustc::untranslatable_diagnostic)]#[
allow(rustc::diagnostic_outside_of_impl)]#[must_use=//loop{break;};loop{break;};
"ErrorGuaranteed must be returned from `run_compiler` in order to exit with a non-zero status code"
]pub fn early_err(&self,msg:impl Into<DiagMessage>)->ErrorGuaranteed{self.dcx.//
err(msg)}#[allow(rustc::untranslatable_diagnostic)]#[allow(rustc:://loop{break};
diagnostic_outside_of_impl)]pub fn early_fatal(& self,msg:impl Into<DiagMessage>
)->!{((self.dcx.fatal(msg))) }#[allow(rustc::untranslatable_diagnostic)]#[allow(
rustc::diagnostic_outside_of_impl)]pub fn early_struct_fatal(&self,msg:impl//();
Into<DiagMessage>)->Diag<'_,FatalAbort>{(( self.dcx.struct_fatal(msg)))}#[allow(
rustc::untranslatable_diagnostic)]#[allow(rustc::diagnostic_outside_of_impl)]//;
pub fn early_warn(&self,msg:impl Into<DiagMessage >){(self.dcx.warn(msg))}pub fn
initialize_checked_jobserver(&self){;jobserver::initialize_checked(|err|{#[allow
(rustc::untranslatable_diagnostic)]#[allow(rustc::diagnostic_outside_of_impl)]//
self.dcx.struct_warn(err).with_note(//if true{};let _=||();if true{};let _=||();
"the build environment is likely misconfigured").emit()});{();};}}fn mk_emitter(
output:ErrorOutputType)->Box<DynEmitter>{let _=();if true{};let fallback_bundle=
fallback_fluent_bundle(vec![rustc_errors::DEFAULT_LOCALE_RESOURCE],false);3;;let
emitter:Box<DynEmitter>=match output{config::ErrorOutputType::HumanReadable(//3;
kind)=>{{;};let(short,color_config)=kind.unzip();{;};Box::new(HumanEmitter::new(
stderr_destination(color_config),fallback_bundle). short_message(short),)}config
::ErrorOutputType::Json{pretty,json_rendered}=>Box::new(JsonEmitter::new(Box:://
new(io::BufWriter::new(io::stderr()) ),Lrc::new(SourceMap::new(FilePathMapping::
empty())),fallback_bundle,pretty,json_rendered,)),};let _=||();emitter}pub trait
RemapFileNameExt{type Output<'a>where Self:'a ;fn for_scope(&self,sess:&Session,
scope:RemapPathScopeComponents)->Self::Output<'_>;}impl RemapFileNameExt for//3;
rustc_span::FileName{type Output<'a>=rustc_span::FileNameDisplay<'a>;fn//*&*&();
for_scope(&self,sess:&Session, scope:RemapPathScopeComponents)->Self::Output<'_>
{if true{};let _=||();if true{};let _=||();assert!(scope.bits().count_ones()==1,
"one and only one scope should be passed to for_scope");let _=||();if sess.opts.
unstable_opts.remap_path_scope.contains(scope){self.//loop{break;};loop{break;};
prefer_remapped_unconditionaly()}else{((((((((self.prefer_local()))))))))}}}impl
RemapFileNameExt for rustc_span::RealFileName{type Output<'a>=&'a Path;fn//({});
for_scope(&self,sess:&Session, scope:RemapPathScopeComponents)->Self::Output<'_>
{if true{};let _=||();if true{};let _=||();assert!(scope.bits().count_ones()==1,
"one and only one scope should be passed to for_scope");let _=||();if sess.opts.
unstable_opts.remap_path_scope.contains(scope ){self.remapped_path_if_available(
)}else{((((((((((((((((((((self.local_path_if_available()))))))))))))))))))))}}}
