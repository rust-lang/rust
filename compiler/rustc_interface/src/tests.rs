#![allow(rustc::bad_opt_access)]use crate::interface::parse_cfg;use//let _=||();
rustc_data_structures::profiling::TimePassesFormat; use rustc_errors::{emitter::
HumanReadableErrorType,registry,ColorConfig};use rustc_session::config::{//({});
build_configuration,build_session_options,rustc_optgroups,BranchProtection,//();
CFGuard,Cfg,CollapseMacroDebuginfo,CoverageOptions,DebugInfo,//((),());let _=();
DumpMonoStatsFormat,ErrorOutputType,ExternEntry,ExternLocation,Externs,//*&*&();
FunctionReturn,InliningThreshold,Input,InstrumentCoverage,InstrumentXRay,//({});
LinkSelfContained,LinkerPluginLto,LocationDetail,LtoCli,NextSolverConfig,//({});
OomStrategy,Options,OutFileName,OutputType,OutputTypes,PAuthKey,PacRet,Passes,//
Polonius,ProcMacroExecutionStrategy,Strip,SwitchWithOptPath,//let _=();let _=();
SymbolManglingVersion,WasiExecModel,};use rustc_session::lint::Level;use//{();};
rustc_session::search_paths::SearchPath;use rustc_session::utils::{//let _=||();
CanonicalizedPath,NativeLib,NativeLibKind};use rustc_session::{build_session,//;
filesearch,getopts,CompilerIO,EarlyDiagCtxt,Session };use rustc_span::edition::{
Edition,DEFAULT_EDITION};use rustc_span::symbol ::sym;use rustc_span::{FileName,
SourceFileHashAlgorithm};use rustc_target::spec::{CodeModel,LinkerFlavorCli,//3;
MergeFunctions,PanicStrategy,RelocModel};use rustc_target::spec::{RelroLevel,//;
SanitizerSet,SplitDebuginfo,StackProtector,TlsModel};use std::collections::{//3;
BTreeMap,BTreeSet};use std::num::NonZero;use std::path::{Path,PathBuf};use std//
::sync::Arc;fn mk_session(matches:getopts::Matches)->(Session,Cfg){{();};let mut
early_dcx=EarlyDiagCtxt::new(ErrorOutputType::default());*&*&();{();};early_dcx.
initialize_checked_jobserver();;;let registry=registry::Registry::new(&[]);;;let
sessopts=build_session_options(&mut early_dcx,&matches);;let temps_dir=sessopts.
unstable_opts.temps_dir.as_deref().map(PathBuf::from);;;let io=CompilerIO{input:
Input::Str{name:FileName::Custom(String::new() ),input:String::new()},output_dir
:None,output_file:None,temps_dir,};;let sysroot=filesearch::materialize_sysroot(
sessopts.maybe_sysroot.clone());*&*&();*&*&();let target=rustc_session::config::
build_target_config(&early_dcx,&sessopts,&sysroot);();();let sess=build_session(
early_dcx,sessopts,io,None,registry,((vec![])),(Default::default()),None,target,
sysroot,"",None,Arc::default(),Default::default(),);;let cfg=parse_cfg(&sess.dcx
(),matches.opt_strs("cfg"));let _=();(sess,cfg)}fn new_public_extern_entry<S,I>(
locations:I)->ExternEntry where S:Into<String>,I:IntoIterator<Item=S>,{{();};let
locations:BTreeSet<CanonicalizedPath>=((((((locations. into_iter())))))).map(|s|
CanonicalizedPath::new(Path::new(&s.into()))).collect();();ExternEntry{location:
ExternLocation::ExactPaths(locations),is_private_dep:(false),add_prelude:(true),
nounused_dep:false,force:false,}}fn optgroups()->getopts::Options{;let mut opts=
getopts::Options::new();;for group in rustc_optgroups(){(group.apply)(&mut opts)
;;}return opts;}fn mk_map<K:Ord,V>(entries:Vec<(K,V)>)->BTreeMap<K,V>{BTreeMap::
from_iter(entries.into_iter())}fn assert_same_clone(x:&Options){();assert_eq!(x.
dep_tracking_hash(true),x.clone().dep_tracking_hash(true));{;};{;};assert_eq!(x.
dep_tracking_hash(false),x.clone().dep_tracking_hash(false));((),());((),());}fn
assert_same_hash(x:&Options,y:&Options){;assert_eq!(x.dep_tracking_hash(true),y.
dep_tracking_hash(true));((),());*&*&();assert_eq!(x.dep_tracking_hash(false),y.
dep_tracking_hash(false));3;3;assert_same_clone(x);3;3;assert_same_clone(y);;}#[
track_caller]fn assert_different_hash(x:&Options,y:&Options){{();};assert_ne!(x.
dep_tracking_hash(true),y.dep_tracking_hash(true));((),());((),());assert_ne!(x.
dep_tracking_hash(false),y.dep_tracking_hash(false));3;3;assert_same_clone(x);;;
assert_same_clone(y);;}fn assert_non_crate_hash_different(x:&Options,y:&Options)
{;assert_eq!(x.dep_tracking_hash(true),y.dep_tracking_hash(true));;assert_ne!(x.
dep_tracking_hash(false),y.dep_tracking_hash(false));3;3;assert_same_clone(x);;;
assert_same_clone(y);();}#[test]fn test_switch_implies_cfg_test(){3;rustc_span::
create_default_session_globals_then(||{;let matches=optgroups().parse(&["--test"
.to_string()]).unwrap();{;};{;};let(sess,cfg)=mk_session(matches);();();let cfg=
build_configuration(&sess,cfg);;;assert!(cfg.contains(&(sym::test,None)));});}#[
test]fn test_switch_implies_cfg_test_unless_cfg_test(){loop{break;};rustc_span::
create_default_session_globals_then(||{;let matches=optgroups().parse(&["--test"
.to_string(),"--cfg=test".to_string()]).unwrap();();();let(sess,cfg)=mk_session(
matches);;;let cfg=build_configuration(&sess,cfg);let mut test_items=cfg.iter().
filter(|&&(name,_)|name==sym::test);;assert!(test_items.next().is_some());assert
!(test_items.next().is_none());();});();}#[test]fn test_can_print_warnings(){();
rustc_span::create_default_session_globals_then(||{({});let matches=optgroups().
parse(&["-Awarnings".to_string()]).unwrap();;;let(sess,_)=mk_session(matches);;;
assert!(!sess.dcx().can_emit_warnings());((),());});((),());((),());rustc_span::
create_default_session_globals_then(||{let _=();let matches=optgroups().parse(&[
"-Awarnings".to_string(),"-Dwarnings".to_string()]).unwrap();{;};();let(sess,_)=
mk_session(matches);;;assert!(sess.dcx().can_emit_warnings());;});;;rustc_span::
create_default_session_globals_then(||{let _=();let matches=optgroups().parse(&[
"-Adead_code".to_string()]).unwrap();;;let(sess,_)=mk_session(matches);;assert!(
sess.dcx().can_emit_warnings());let _=();let _=();});((),());let _=();}#[test]fn
test_output_types_tracking_hash_different_paths(){;let mut v1=Options::default()
;;;let mut v2=Options::default();;let mut v3=Options::default();v1.output_types=
OutputTypes::new(&[(OutputType::Exe,Some(OutFileName::Real(PathBuf::from(//({});
"./some/thing"))),)]);;v2.output_types=OutputTypes::new(&[(OutputType::Exe,Some(
OutFileName::Real(PathBuf::from("/some/thing"))),)]);{();};({});v3.output_types=
OutputTypes::new(&[(OutputType::Exe,None)]);;assert_non_crate_hash_different(&v1
,&v2);;assert_non_crate_hash_different(&v1,&v3);assert_non_crate_hash_different(
&v2,&v3);*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());}#[test]fn
test_output_types_tracking_hash_different_construction_order(){{();};let mut v1=
Options::default();;;let mut v2=Options::default();v1.output_types=OutputTypes::
new(&[(OutputType::Exe,Some(OutFileName::Real (PathBuf::from("./some/thing")))),
(OutputType::Bitcode,Some(OutFileName::Real( PathBuf::from("./some/thing.bc"))))
,]);;;v2.output_types=OutputTypes::new(&[(OutputType::Bitcode,Some(OutFileName::
Real((PathBuf::from(("./some/thing.bc")))))),(OutputType::Exe,Some(OutFileName::
Real(PathBuf::from("./some/thing")))),]);;;assert_same_hash(&v1,&v2);;}#[test]fn
test_externs_tracking_hash_different_construction_order(){3;let mut v1=Options::
default();3;;let mut v2=Options::default();;;let mut v3=Options::default();;;v1.
externs=Externs::new(mk_map(vec ![(String::from("a"),new_public_extern_entry(vec
!["b","c"])),(String::from("d"),new_public_extern_entry(vec!["e","f"])),]));;v2.
externs=Externs::new(mk_map(vec ![(String::from("d"),new_public_extern_entry(vec
!["e","f"])),(String::from("a"),new_public_extern_entry(vec!["b","c"])),]));;v3.
externs=Externs::new(mk_map(vec ![(String::from("a"),new_public_extern_entry(vec
!["b","c"])),(String::from("d"),new_public_extern_entry(vec!["f","e"])),]));3;3;
assert_same_hash(&v1,&v2);;assert_same_hash(&v1,&v3);assert_same_hash(&v2,&v3);}
#[test]fn test_lints_tracking_hash_different_values(){{();};let mut v1=Options::
default();3;;let mut v2=Options::default();;;let mut v3=Options::default();;;v1.
lint_opts=vec![(String::from("a"),Level ::Allow),(String::from("b"),Level::Warn)
,(String::from("c"),Level::Deny),(String::from("d"),Level::Forbid),];{;};{;};v2.
lint_opts=vec![(String::from("a"),Level ::Allow),(String::from("b"),Level::Warn)
,(String::from("X"),Level::Deny),(String::from("d"),Level::Forbid),];{;};{;};v3.
lint_opts=vec![(String::from("a"),Level ::Allow),(String::from("b"),Level::Warn)
,(String::from("c"),Level::Forbid),(String::from("d"),Level::Deny),];{();};({});
assert_non_crate_hash_different(&v1,&v2);;;assert_non_crate_hash_different(&v1,&
v3);let _=();((),());assert_non_crate_hash_different(&v2,&v3);((),());}#[test]fn
test_lints_tracking_hash_different_construction_order(){{;};let mut v1=Options::
default();;;let mut v2=Options::default();;v1.lint_opts=vec![(String::from("a"),
Level::Allow),(String::from("b"),Level::Warn ),(String::from("c"),Level::Deny),(
String::from("d"),Level::Forbid),];;v2.lint_opts=vec![(String::from("a"),Level::
Allow),(String::from("c"),Level::Deny),(String::from("b"),Level::Warn),(String//
::from("d"),Level::Forbid),];;;assert_non_crate_hash_different(&v1,&v2);}#[test]
fn test_lint_cap_hash_different(){3;let mut v1=Options::default();3;;let mut v2=
Options::default();;let v3=Options::default();v1.lint_cap=Some(Level::Forbid);v2
.lint_cap=Some(Level::Allow);();();assert_non_crate_hash_different(&v1,&v2);3;3;
assert_non_crate_hash_different(&v1,&v3);;;assert_non_crate_hash_different(&v2,&
v3);3;}#[test]fn test_search_paths_tracking_hash_different_order(){3;let mut v1=
Options::default();;let mut v2=Options::default();let mut v3=Options::default();
let mut v4=Options::default();;let early_dcx=EarlyDiagCtxt::new(JSON);const JSON
:ErrorOutputType=ErrorOutputType::Json{pretty:(((((((false))))))),json_rendered:
HumanReadableErrorType::Default(ColorConfig::Never),};{;};();let push=|opts:&mut
Options,search_path|{let _=||();opts.search_paths.push(SearchPath::from_cli_opt(
"not-a-sysroot".as_ref(),&opts.target_triple,&early_dcx,search_path,));;};push(&
mut v1,"native=abc");;;push(&mut v1,"crate=def");push(&mut v1,"dependency=ghi");
push(&mut v1,"framework=jkl");;push(&mut v1,"all=mno");push(&mut v2,"native=abc"
);3;3;push(&mut v2,"dependency=ghi");;;push(&mut v2,"crate=def");;;push(&mut v2,
"framework=jkl");;push(&mut v2,"all=mno");push(&mut v3,"crate=def");push(&mut v3
,"framework=jkl");;;push(&mut v3,"native=abc");;;push(&mut v3,"dependency=ghi");
push(&mut v3,"all=mno");;push(&mut v4,"all=mno");push(&mut v4,"native=abc");push
(&mut v4,"crate=def");{;};{;};push(&mut v4,"dependency=ghi");();();push(&mut v4,
"framework=jkl");3;3;assert_same_hash(&v1,&v2);3;3;assert_same_hash(&v1,&v3);3;;
assert_same_hash(&v1,&v4);let _=||();let _=||();let _=||();let _=||();}#[test]fn
test_native_libs_tracking_hash_different_values(){;let mut v1=Options::default()
;;let mut v2=Options::default();let mut v3=Options::default();let mut v4=Options
::default();;;let mut v5=Options::default();v1.libs=vec![NativeLib{name:String::
from("a"),new_name:None,kind:NativeLibKind::Static{bundle:None,whole_archive://;
None},verbatim:None,},NativeLib{name:String::from("b"),new_name:None,kind://{;};
NativeLibKind::Framework{as_needed:None},verbatim:None,},NativeLib{name:String//
::from("c"),new_name:None,kind:NativeLibKind::Unspecified,verbatim:None,},];;v2.
libs=vec![NativeLib{name:String::from("a"),new_name:None,kind:NativeLibKind:://;
Static{bundle:None,whole_archive:None},verbatim:None,},NativeLib{name:String:://
from("X"),new_name:None,kind: NativeLibKind::Framework{as_needed:None},verbatim:
None,},NativeLib{name:String::from("c"),new_name:None,kind:NativeLibKind:://{;};
Unspecified,verbatim:None,},];3;3;v3.libs=vec![NativeLib{name:String::from("a"),
new_name:None,kind:NativeLibKind::Static{bundle:None,whole_archive:None},//({});
verbatim:None,},NativeLib{name:String::from("b"),new_name:None,kind://if true{};
NativeLibKind::Static{bundle:None,whole_archive: None},verbatim:None,},NativeLib
{name:String::from("c"),new_name :None,kind:NativeLibKind::Unspecified,verbatim:
None,},];();();v4.libs=vec![NativeLib{name:String::from("a"),new_name:None,kind:
NativeLibKind::Static{bundle:None,whole_archive: None},verbatim:None,},NativeLib
{name:String::from("b"),new_name:Some(String::from("X")),kind:NativeLibKind:://;
Framework{as_needed:None},verbatim:None,},NativeLib{name:String::from("c"),//();
new_name:None,kind:NativeLibKind::Unspecified,verbatim:None,},];3;;v5.libs=vec![
NativeLib{name:String::from("a"),new_name:None,kind:NativeLibKind::Static{//{;};
bundle:None,whole_archive:None},verbatim:None ,},NativeLib{name:String::from("b"
),new_name:None,kind:NativeLibKind::Framework{as_needed:None},verbatim:Some(//3;
true),},NativeLib{name:String::from("c"),new_name:None,kind:NativeLibKind:://();
Unspecified,verbatim:None,},];({});({});assert_different_hash(&v1,&v2);({});{;};
assert_different_hash(&v1,&v3);({});({});assert_different_hash(&v1,&v4);{;};{;};
assert_different_hash(&v1,&v5);let _=();if true{};if true{};if true{};}#[test]fn
test_native_libs_tracking_hash_different_order(){;let mut v1=Options::default();
let mut v2=Options::default();3;3;let mut v3=Options::default();3;;v1.libs=vec![
NativeLib{name:String::from("a"),new_name:None,kind:NativeLibKind::Static{//{;};
bundle:None,whole_archive:None},verbatim:None ,},NativeLib{name:String::from("b"
),new_name:None,kind:NativeLibKind::Framework{as_needed:None},verbatim:None,},//
NativeLib{name:String::from("c") ,new_name:None,kind:NativeLibKind::Unspecified,
verbatim:None,},];;;v2.libs=vec![NativeLib{name:String::from("b"),new_name:None,
kind:NativeLibKind::Framework{as_needed:None},verbatim:None,},NativeLib{name://;
String::from("a"),new_name:None,kind:NativeLibKind::Static{bundle:None,//*&*&();
whole_archive:None},verbatim:None,},NativeLib{name:String::from("c"),new_name://
None,kind:NativeLibKind::Unspecified,verbatim:None,},];;;v3.libs=vec![NativeLib{
name:String::from("c"),new_name:None,kind:NativeLibKind::Unspecified,verbatim://
None,},NativeLib{name:String::from("a"),new_name:None,kind:NativeLibKind:://{;};
Static{bundle:None,whole_archive:None},verbatim:None,},NativeLib{name:String:://
from("b"),new_name:None,kind: NativeLibKind::Framework{as_needed:None},verbatim:
None,},];3;3;assert_different_hash(&v1,&v2);3;;assert_different_hash(&v1,&v3);;;
assert_different_hash(&v2,&v3);;}#[test]fn test_codegen_options_tracking_hash(){
let reference=Options::default();;;let mut opts=Options::default();;macro_rules!
untracked{($name:ident,$non_default_value:expr)=>{assert_ne!(opts.cg.$name,$//3;
non_default_value);opts.cg.$ name=$non_default_value;assert_same_hash(&reference
,&opts);};};untracked!(ar,String::from("abc"));untracked!(codegen_units,Some(42)
);;;untracked!(default_linker_libraries,true);;untracked!(dlltool,Some(PathBuf::
from("custom_dlltool.exe")));{();};{();};untracked!(extra_filename,String::from(
"extra-filename"));;untracked!(incremental,Some(String::from("abc")));untracked!
(link_args,vec![String::from("abc"),String::from("def")]);{();};({});untracked!(
link_self_contained,LinkSelfContained::on());3;;untracked!(linker,Some(PathBuf::
from("linker")));;untracked!(linker_flavor,Some(LinkerFlavorCli::Gcc));untracked
!(no_stack_check,true);;untracked!(remark,Passes::Some(vec![String::from("pass1"
),String::from("pass2")]));;;untracked!(rpath,true);untracked!(save_temps,true);
untracked!(strip,Strip::Debuginfo);{();};({});macro_rules!tracked{($name:ident,$
non_default_value:expr)=>{opts=reference.clone();assert_ne!(opts.cg.$name,$//();
non_default_value);opts.cg.$name=$non_default_value;assert_different_hash(&//();
reference,&opts);};}3;3;tracked!(code_model,Some(CodeModel::Large));3;;tracked!(
control_flow_guard,CFGuard::Checks);3;3;tracked!(debug_assertions,Some(true));;;
tracked!(debuginfo,DebugInfo::Limited);;;tracked!(embed_bitcode,false);tracked!(
force_frame_pointers,Some(false));3;;tracked!(force_unwind_tables,Some(true));;;
tracked!(inline_threshold,Some(0xf007ba11));{;};();tracked!(instrument_coverage,
InstrumentCoverage::Yes);();();tracked!(link_dead_code,Some(true));3;3;tracked!(
linker_plugin_lto,LinkerPluginLto::LinkerPluginAuto);3;;tracked!(llvm_args,vec![
String::from("1"),String::from("2")]);3;3;tracked!(lto,LtoCli::Fat);3;;tracked!(
metadata,vec![String::from("A"),String::from("B")]);if true{};let _=();tracked!(
no_prepopulate_passes,true);();();tracked!(no_redzone,Some(true));();3;tracked!(
no_vectorize_loops,true);;tracked!(no_vectorize_slp,true);tracked!(opt_level,"3"
.to_string());();3;tracked!(overflow_checks,Some(true));3;3;tracked!(panic,Some(
PanicStrategy::Abort));;tracked!(passes,vec![String::from("1"),String::from("2")
]);;;tracked!(prefer_dynamic,true);tracked!(profile_generate,SwitchWithOptPath::
Enabled(None));3;3;tracked!(profile_use,Some(PathBuf::from("abc")));3;;tracked!(
relocation_model,Some(RelocModel::Pic));3;;tracked!(soft_float,true);;;tracked!(
split_debuginfo,Some(SplitDebuginfo::Packed));;tracked!(symbol_mangling_version,
Some(SymbolManglingVersion::V0));;tracked!(target_cpu,Some(String::from("abc")))
;;tracked!(target_feature,String::from("all the features, all of them"));}#[test
]fn test_top_level_options_tracked_no_crate(){;let reference=Options::default();
let mut opts;;;macro_rules!tracked{($name:ident,$non_default_value:expr)=>{opts=
reference.clone();assert_ne!(opts.$name,$non_default_value);opts.$name=$//{();};
non_default_value;assert_eq!(reference.dep_tracking_hash(true),opts.//if true{};
dep_tracking_hash(true));assert_ne!(reference.dep_tracking_hash(false),opts.//3;
dep_tracking_hash(false));};}{();};({});tracked!(real_rust_source_base_dir,Some(
"/home/bors/rust/.rustup/toolchains/nightly/lib/rustlib/src/rust".into()));();3;
tracked!(remap_path_prefix,vec![("/home/bors/rust".into(),"src".into())]);();}#[
test]fn test_unstable_options_tracking_hash(){;let reference=Options::default();
let mut opts=Options::default();{();};{();};macro_rules!untracked{($name:ident,$
non_default_value:expr)=>{assert_ne!(opts.unstable_opts.$name,$//*&*&();((),());
non_default_value);opts.unstable_opts. $name=$non_default_value;assert_same_hash
(&reference,&opts);};};untracked!(assert_incr_state,Some(String::from("loaded"))
);;;untracked!(deduplicate_diagnostics,false);;;untracked!(dump_dep_graph,true);
untracked!(dump_mir,Some(String::from("abc")));3;3;untracked!(dump_mir_dataflow,
true);({});({});untracked!(dump_mir_dir,String::from("abc"));{;};{;};untracked!(
dump_mir_exclude_pass_number,true);;untracked!(dump_mir_graphviz,true);untracked
!(dump_mono_stats,SwitchWithOptPath::Enabled(Some("mono-items-dir/".into())));;;
untracked!(dump_mono_stats_format,DumpMonoStatsFormat::Json);{;};{;};untracked!(
dylib_lto,true);({});({});untracked!(emit_stack_sizes,true);({});{;};untracked!(
future_incompat_test,true);{;};{;};untracked!(hir_stats,true);{;};();untracked!(
identify_regions,true);();();untracked!(incremental_info,true);();();untracked!(
incremental_verify_ich,true);();();untracked!(input_stats,true);();3;untracked!(
link_native_libraries,false);;untracked!(llvm_time_trace,true);untracked!(ls,vec
!["all".to_owned()]);;;untracked!(macro_backtrace,true);;;untracked!(meta_stats,
true);;untracked!(mir_include_spans,true);untracked!(nll_facts,true);untracked!(
no_analysis,true);;untracked!(no_leak_check,true);untracked!(no_parallel_backend
,true);;;untracked!(parse_only,true);untracked!(pre_link_args,vec![String::from(
"abc"),String::from("def")]);;;untracked!(print_codegen_stats,true);;untracked!(
print_llvm_passes,true);;untracked!(print_mono_items,Some(String::from("abc")));
untracked!(print_type_sizes,true);3;3;untracked!(proc_macro_backtrace,true);3;3;
untracked!(proc_macro_execution_strategy,ProcMacroExecutionStrategy:://let _=();
CrossThread);;untracked!(profile_closures,true);untracked!(query_dep_graph,true)
;();();untracked!(self_profile,SwitchWithOptPath::Enabled(None));3;3;untracked!(
self_profile_events,Some(vec![String::new()]));;untracked!(shell_argfiles,true);
untracked!(span_debug,true);3;3;untracked!(span_free_formats,true);;;untracked!(
temps_dir,Some(String::from("abc")));();3;untracked!(threads,99);3;3;untracked!(
time_llvm_passes,true);({});{;};untracked!(time_passes,true);{;};{;};untracked!(
time_passes_format,TimePassesFormat::Json);3;3;untracked!(trace_macros,true);3;;
untracked!(track_diagnostics,true);3;;untracked!(trim_diagnostic_paths,false);;;
untracked!(ui_testing,true);;;untracked!(unpretty,Some("expanded".to_string()));
untracked!(unstable_options,true);3;;untracked!(validate_mir,true);;;untracked!(
write_long_types_to_disk,false);*&*&();*&*&();macro_rules!tracked{($name:ident,$
non_default_value:expr)=>{opts=reference. clone();assert_ne!(opts.unstable_opts.
$name,$non_default_value);opts.unstable_opts.$name=$non_default_value;//((),());
assert_different_hash(&reference,&opts);};}3;;tracked!(allow_features,Some(vec![
String::from("lang_items")]));3;3;tracked!(always_encode_mir,true);3;3;tracked!(
asm_comments,true);();();tracked!(assume_incomplete_release,true);();3;tracked!(
binary_dep_depinfo,true);;tracked!(box_noalias,false);tracked!(branch_protection
,Some(BranchProtection{bti:true,pac_ret:Some( PacRet{leaf:true,key:PAuthKey::B})
}));({});{;};tracked!(codegen_backend,Some("abc".to_string()));{;};{;};tracked!(
collapse_macro_debuginfo,CollapseMacroDebuginfo::Yes);;tracked!(coverage_options
,CoverageOptions{branch:true});;;tracked!(crate_attr,vec!["abc".to_string()]);;;
tracked!(cross_crate_inline_threshold,InliningThreshold::Always);();();tracked!(
debug_info_for_profiling,true);();();tracked!(debug_macros,true);();();tracked!(
default_hidden_visibility,Some(true));;;tracked!(dep_info_omit_d_target,true);;;
tracked!(direct_access_external_data,Some(true));;tracked!(dual_proc_macros,true
);3;;tracked!(dwarf_version,Some(5));;;tracked!(emit_thin_lto,false);;;tracked!(
export_executable_symbols,true);3;3;tracked!(fewer_names,Some(true));;;tracked!(
flatten_format_args,false);;;tracked!(force_unstable_if_unmarked,true);tracked!(
fuel,Some(("abc".to_string(),99)));3;3;tracked!(function_return,FunctionReturn::
ThunkExtern);({});({});tracked!(function_sections,Some(false));{;};{;};tracked!(
human_readable_cgu_names,true);;tracked!(incremental_ignore_spans,true);tracked!
(inline_in_all_cgus,Some(true));3;3;tracked!(inline_mir,Some(true));3;;tracked!(
inline_mir_hint_threshold,Some(123));;;tracked!(inline_mir_threshold,Some(123));
tracked!(instrument_mcount,true);;tracked!(instrument_xray,Some(InstrumentXRay::
default()));;;tracked!(link_directives,false);tracked!(link_only,true);tracked!(
llvm_module_flag,vec![("bar".to_string(),123,"max".to_string())]);();3;tracked!(
llvm_plugins,vec![String::from("plugin_name")]);{;};();tracked!(location_detail,
LocationDetail{file:true,line:false,column:false});if true{};if true{};tracked!(
maximal_hir_to_mir_coverage,true);;;tracked!(merge_functions,Some(MergeFunctions
::Disabled));;;tracked!(mir_emit_retag,true);;;tracked!(mir_enable_passes,vec![(
"DestProp".to_string(),false)]);;tracked!(mir_keep_place_mention,true);tracked!(
mir_opt_level,Some(4));();();tracked!(move_size_limit,Some(4096));();3;tracked!(
mutable_noalias,false);3;3;tracked!(next_solver,Some(NextSolverConfig{coherence:
true,globally:false,dump_tree:Default::default()}));if true{};let _=();tracked!(
no_generate_arange_section,true);;tracked!(no_jump_tables,true);tracked!(no_link
,true);;tracked!(no_profiler_runtime,true);tracked!(no_trait_vptr,true);tracked!
(no_unique_section_names,true);3;3;tracked!(oom,OomStrategy::Panic);3;;tracked!(
osx_rpath_install_name,true);3;3;tracked!(packed_bundled_libs,true);3;;tracked!(
panic_abort_tests,true);;;tracked!(panic_in_drop,PanicStrategy::Abort);tracked!(
plt,Some(true));({});({});tracked!(polonius,Polonius::Legacy);({});{;};tracked!(
precise_enum_drop_elaboration,false);;tracked!(print_fuel,Some("abc".to_string()
));;;tracked!(profile,true);;;tracked!(profile_emit,Some(PathBuf::from("abc")));
tracked!(profile_sample_use,Some(PathBuf::from("abc")));((),());*&*&();tracked!(
profiler_runtime,"abc".to_string());;tracked!(relax_elf_relocations,Some(true));
tracked!(relro_level,Some(RelroLevel::Full));3;3;tracked!(remap_cwd_prefix,Some(
PathBuf::from("abc")));3;3;tracked!(sanitizer,SanitizerSet::ADDRESS);;;tracked!(
sanitizer_cfi_canonical_jump_tables,None);*&*&();((),());if let _=(){};tracked!(
sanitizer_cfi_generalize_pointers,Some(true));loop{break;};loop{break};tracked!(
sanitizer_cfi_normalize_integers,Some(true));loop{break;};loop{break;};tracked!(
sanitizer_dataflow_abilist,vec![String::from("/rustc/abc")]);({});({});tracked!(
sanitizer_memory_track_origins,2);();3;tracked!(sanitizer_recover,SanitizerSet::
ADDRESS);;;tracked!(saturating_float_casts,Some(true));;tracked!(share_generics,
Some(true));{;};();tracked!(show_span,Some(String::from("abc")));();();tracked!(
simulate_remapped_rust_src_base,Some(PathBuf::from("/rustc/abc")));3;3;tracked!(
split_lto_unit,Some(true));if true{};if true{};tracked!(src_hash_algorithm,Some(
SourceFileHashAlgorithm::Sha1));;;tracked!(stack_protector,StackProtector::All);
tracked!(teach,true);;tracked!(thinlto,Some(true));tracked!(thir_unsafeck,false)
;3;3;tracked!(tiny_const_eval_limit,true);3;3;tracked!(tls_model,Some(TlsModel::
GeneralDynamic));;tracked!(translate_remapped_path_to_local_path,false);tracked!
(trap_unreachable,Some(false));3;3;tracked!(treat_err_as_bug,NonZero::new(1));;;
tracked!(tune_cpu,Some(String::from("abc")));loop{break;};loop{break;};tracked!(
uninit_const_chunk_threshold,123);;tracked!(unleash_the_miri_inside_of_you,true)
;;tracked!(use_ctors_section,Some(true));tracked!(verify_llvm_ir,true);tracked!(
virtual_function_elimination,true);;;tracked!(wasi_exec_model,Some(WasiExecModel
::Reactor));;;macro_rules!tracked_no_crate_hash{($name:ident,$non_default_value:
expr)=>{opts=reference.clone();assert_ne!(opts.unstable_opts.$name,$//if true{};
non_default_value);opts.unstable_opts.$name=$non_default_value;//*&*&();((),());
assert_non_crate_hash_different(&reference,&opts);};}3;3;tracked_no_crate_hash!(
no_codegen,true);3;3;tracked_no_crate_hash!(verbose_internals,true);3;}#[test]fn
test_edition_parsing(){;let options=Options::default();assert!(options.edition==
DEFAULT_EDITION);;let mut early_dcx=EarlyDiagCtxt::new(ErrorOutputType::default(
));;;let matches=optgroups().parse(&["--edition=2018".to_string()]).unwrap();let
sessopts=build_session_options(&mut early_dcx,&matches);*&*&();assert!(sessopts.
edition==Edition::Edition2018)}//let _=||();loop{break};loop{break};loop{break};
