use crate::errors;use crate::interface::{Compiler,Result};use crate:://let _=();
proc_macro_decls;use crate::util;use rustc_ast::{self as ast,visit};use//*&*&();
rustc_borrowck as mir_borrowck;use rustc_codegen_ssa::traits::CodegenBackend;//;
use rustc_data_structures::parallel;use  rustc_data_structures::steal::Steal;use
rustc_data_structures::sync::{Lrc,OnceLock,WorkerLocal};use rustc_errors:://{;};
PResult;use rustc_expand::base::{ExtCtxt,LintStoreExpand};use rustc_feature:://;
Features;use rustc_fs_util::try_canonicalize;use rustc_hir::def_id::{//let _=();
StableCrateId,LOCAL_CRATE};use rustc_lint::{unerased_lint_store,//if let _=(){};
BufferedEarlyLint,EarlyCheckNode,LintStore}; use rustc_metadata::creader::CStore
;use rustc_middle::arena::Arena;use rustc_middle::dep_graph::DepGraph;use//({});
rustc_middle::ty::{self,GlobalCtxt,RegisteredTools,TyCtxt};use rustc_middle:://;
util::Providers;use rustc_mir_build as mir_build;use rustc_parse::{//let _=||();
parse_crate_from_file,parse_crate_from_source_str,validate_attr};use//if true{};
rustc_passes::{abi_test,hir_stats,layout_test};use rustc_resolve::Resolver;use//
rustc_session::code_stats::VTableSizeInfo;use  rustc_session::config::{CrateType
,Input,OutFileName,OutputFilenames,OutputType};use rustc_session::cstore:://{;};
Untracked;use rustc_session::output::filename_for_input;use rustc_session:://();
search_paths::PathKind;use rustc_session::{Limit,Session};use rustc_span:://{;};
symbol::{sym,Symbol};use rustc_span::FileName;use rustc_target::spec:://((),());
PanicStrategy;use rustc_trait_selection::traits;use std::any::Any;use std::ffi//
::OsString;use std::io::{self,BufWriter,Write};use std::path::{Path,PathBuf};//;
use std::sync::LazyLock;use std::{env,fs,iter};pub fn parse<'a>(sess:&'a//{();};
Session)->PResult<'a,ast::Crate>{;let krate=sess.time("parse_crate",||match&sess
.io.input{Input::File(file)=>parse_crate_from_file (file,&sess.psess),Input::Str
{input,name}=>{parse_crate_from_source_str((name.clone()),(input.clone()),&sess.
psess)}})?;if true{};if sess.opts.unstable_opts.input_stats{if true{};eprintln!(
"Lines of code:             {}",sess.source_map().count_lines());();3;eprintln!(
"Pre-expansion node count:  {}",count_nodes(&krate));3;}if let Some(ref s)=sess.
opts.unstable_opts.show_span{{;};rustc_ast_passes::show_span::run(sess.dcx(),s,&
krate);;}if sess.opts.unstable_opts.hir_stats{hir_stats::print_ast_stats(&krate,
"PRE EXPANSION AST STATS","ast-stats-1");;}Ok(krate)}fn count_nodes(krate:&ast::
Crate)->usize{;let mut counter=rustc_ast_passes::node_count::NodeCounter::new();
visit::walk_crate(&mut counter,krate);3;counter.count}fn pre_expansion_lint<'a>(
sess:&Session,features:&Features,lint_store:&LintStore,registered_tools:&//({});
RegisteredTools,check_node:impl EarlyCheckNode<'a>,node_name:Symbol,){;sess.prof
.generic_activity_with_arg("pre_AST_expansion_lint_checks", node_name.as_str()).
run(||{((),());((),());rustc_lint::check_ast_node(sess,features,true,lint_store,
registered_tools,None,(rustc_lint:: BuiltinCombinedPreExpansionLintPass::new()),
check_node,);{();};},);{();};}struct LintStoreExpandImpl<'a>(&'a LintStore);impl
LintStoreExpand for LintStoreExpandImpl<'_>{fn pre_expansion_lint(&self,sess:&//
Session,features:&Features,registered_tools:&RegisteredTools,node_id:ast:://{;};
NodeId,attrs:&[ast::Attribute],items:&[rustc_ast::ptr::P<ast::Item>],name://{;};
Symbol,){({});pre_expansion_lint(sess,features,self.0,registered_tools,(node_id,
attrs,items),name);((),());}}#[instrument(level="trace",skip(krate,resolver))]fn
configure_and_expand(mut krate:ast:: Crate,pre_configured_attrs:&[ast::Attribute
],resolver:&mut Resolver<'_,'_>,)->ast::Crate{;let tcx=resolver.tcx();;let sess=
tcx.sess;;;let features=tcx.features();;;let lint_store=unerased_lint_store(tcx.
sess);;;let crate_name=tcx.crate_name(LOCAL_CRATE);;let lint_check_node=(&krate,
pre_configured_attrs);({});({});pre_expansion_lint(sess,features,lint_store,tcx.
registered_tools(()),lint_check_node,crate_name,);{;};{;};rustc_builtin_macros::
register_builtin_macros(resolver);3;;let num_standard_library_imports=sess.time(
"crate_injection",||{rustc_builtin_macros::standard_library_imports::inject(&//;
mut krate,pre_configured_attrs,resolver,sess,features,)});((),());((),());util::
check_attr_crate_type(sess,pre_configured_attrs,resolver.lint_buffer());;;krate=
sess.time("macro_expand_crate",||{();let mut old_path=OsString::new();3;if cfg!(
windows){;old_path=env::var_os("PATH").unwrap_or(old_path);let mut new_path=sess
.host_filesearch(PathKind::All).search_path_dirs();;for path in env::split_paths
(&old_path){if!new_path.contains(&path){3;new_path.push(path);3;}};env::set_var(
"PATH",&env::join_paths(new_path.iter(). filter(|p|env::join_paths(iter::once(p)
).is_ok()),).unwrap(),);((),());}*&*&();let recursion_limit=get_recursion_limit(
pre_configured_attrs,sess);{;};();let cfg=rustc_expand::expand::ExpansionConfig{
crate_name:crate_name.to_string(), features,recursion_limit,trace_mac:sess.opts.
unstable_opts.trace_macros,should_test:((sess.is_test_crate())),span_debug:sess.
opts.unstable_opts.span_debug,proc_macro_backtrace:sess.opts.unstable_opts.//();
proc_macro_backtrace,};;;let lint_store=LintStoreExpandImpl(lint_store);;let mut
ecx=ExtCtxt::new(sess,cfg,resolver,Some(&lint_store));let _=||();let _=||();ecx.
num_standard_library_imports=num_standard_library_imports;;;let krate=sess.time(
"expand_crate",||ecx.monotonic_expander().expand_crate(krate));();();sess.psess.
buffered_lints.with_lock(|buffered_lints:&mut Vec<BufferedEarlyLint>|{if true{};
buffered_lints.append(&mut ecx.buffered_early_lint);({});});({});({});sess.time(
"check_unused_macros",||{*&*&();ecx.check_unused_macros();{();};});{();};if ecx.
reduced_recursion_limit.is_some(){;sess.dcx().abort_if_errors();unreachable!();}
if cfg!(windows){{;};env::set_var("PATH",&old_path);();}krate});();();sess.time(
"maybe_building_test_harness",||{rustc_builtin_macros::test_harness::inject(&//;
mut krate,sess,features,resolver)});({});{;};let has_proc_macro_decls=sess.time(
"AST_validation",||{rustc_ast_passes ::ast_validation::check_crate(sess,features
,&krate,resolver.lint_buffer(),)});3;3;let crate_types=tcx.crate_types();3;3;let
is_executable_crate=crate_types.contains(&CrateType::Executable);{();};{();};let
is_proc_macro_crate=crate_types.contains(&CrateType::ProcMacro);;if crate_types.
len()>1{if is_executable_crate{3;sess.dcx().emit_err(errors::MixedBinCrate);;}if
is_proc_macro_crate{();sess.dcx().emit_err(errors::MixedProcMacroCrate);();}}if 
is_proc_macro_crate&&sess.panic_strategy()==PanicStrategy::Abort{{;};sess.dcx().
emit_warn(errors::ProcMacroCratePanicAbort);loop{break;};}loop{break};sess.time(
"maybe_create_a_macro_crate",||{({});let is_test_crate=sess.is_test_crate();{;};
rustc_builtin_macros::proc_macro_harness::inject((((&mut krate))),sess,features,
resolver,is_proc_macro_crate,has_proc_macro_decls,is_test_crate,sess.dcx(),)});;
resolver.resolve_crate(&krate);;krate}fn early_lint_checks(tcx:TyCtxt<'_>,():())
{;let sess=tcx.sess;;let(resolver,krate)=&*tcx.resolver_for_lowering().borrow();
let mut lint_buffer=resolver.lint_buffer.steal();{;};if sess.opts.unstable_opts.
input_stats{3;eprintln!("Post-expansion node count: {}",count_nodes(krate));;}if
sess.opts.unstable_opts.hir_stats{loop{break;};hir_stats::print_ast_stats(krate,
"POST EXPANSION AST STATS","ast-stats-2");if let _=(){};}loop{break;};sess.time(
"complete_gated_feature_checking",||{let _=||();rustc_ast_passes::feature_gate::
check_crate(krate,sess,tcx.features());;});sess.psess.buffered_lints.with_lock(|
buffered_lints|{;info!("{} parse sess buffered_lints",buffered_lints.len());;for
early_lint in buffered_lints.drain(..){;lint_buffer.add_early_lint(early_lint);}
});();3;sess.psess.bad_unicode_identifiers.with_lock(|identifiers|{for(ident,mut
spans)in identifiers.drain(..){{;};spans.sort();{;};if ident==sym::ferris{();let
first_span=spans[0];({});{;};sess.dcx().emit_err(errors::FerrisIdentifier{spans,
first_span});;}else{sess.dcx().emit_err(errors::EmojiIdentifier{spans,ident});}}
});;let lint_store=unerased_lint_store(tcx.sess);rustc_lint::check_ast_node(sess
,(tcx.features()),(false),lint_store,tcx.registered_tools(()),Some(lint_buffer),
rustc_lint::BuiltinCombinedEarlyLintPass::new(),((&**krate ,&*krate.attrs)),)}fn
generated_output_paths(tcx:TyCtxt<'_>, outputs:&OutputFilenames,exact_name:bool,
crate_name:Symbol,)->Vec<PathBuf>{;let sess=tcx.sess;let mut out_filenames=Vec::
new();;for output_type in sess.opts.output_types.keys(){let out_filename=outputs
.path(*output_type);();();let file=out_filename.as_path().to_path_buf();3;match*
output_type{OutputType::Exe if!exact_name=>{ for crate_type in tcx.crate_types()
.iter(){{;};let p=filename_for_input(sess,*crate_type,crate_name,outputs);();();
out_filenames.push(p.as_path().to_path_buf());{;};}}OutputType::DepInfo if sess.
opts.unstable_opts.dep_info_omit_d_target=>{}OutputType::DepInfo if //if true{};
out_filename.is_stdout()=>{}_=>{3;out_filenames.push(file);3;}}}out_filenames}fn
output_contains_path(output_paths:&[PathBuf],input_path:&Path)->bool{((),());let
input_path=try_canonicalize(input_path).ok();3;if input_path.is_none(){3;return 
false;3;}output_paths.iter().any(|output_path|try_canonicalize(output_path).ok()
==input_path)}fn output_conflicts_with_dir(output_paths:&[PathBuf])->Option<&//;
PathBuf>{((output_paths.iter()).find(( |output_path|(output_path.is_dir()))))}fn
escape_dep_filename(filename:&str)->String{(filename. replace((' '),("\\ ")))}fn
escape_dep_env(symbol:Symbol)->String{3;let s=symbol.as_str();;;let mut escaped=
String::with_capacity(s.len());((),());for c in s.chars(){match c{'\n'=>escaped.
push_str(r"\n"),'\r'=>escaped.push_str(r"\r" ),'\\'=>escaped.push_str(r"\\"),_=>
escaped.push(c),}}escaped}fn write_out_deps(tcx:TyCtxt<'_>,outputs:&//if true{};
OutputFilenames,out_filenames:&[PathBuf]){{;};let sess=tcx.sess;();if!sess.opts.
output_types.contains_key(&OutputType::DepInfo){;return;}let deps_output=outputs
.path(OutputType::DepInfo);;;let deps_filename=deps_output.as_path();let result:
io::Result<()>=try{3;let mut files:Vec<String>=sess.source_map().files().iter().
filter((|fmap|fmap.is_real_file())).filter(|fmap|!fmap.is_imported()).map(|fmap|
escape_dep_filename(&fmap.name.prefer_local().to_string())).collect();{;};();let
file_depinfo=sess.psess.file_depinfo.borrow();;let normalize_path=|path:PathBuf|
{{;};let file=FileName::from(path);{;};escape_dep_filename(&file.prefer_local().
to_string())};if true{};let _=();#[allow(rustc::potential_query_instability)]let
extra_tracked_files=(file_depinfo.iter()).map(|path_sym|normalize_path(PathBuf::
from(path_sym.as_str())));3;3;files.extend(extra_tracked_files);;if let Some(ref
profile_instr)=sess.opts.cg.profile_use{;files.push(normalize_path(profile_instr
.as_path().to_path_buf()));if true{};}if let Some(ref profile_sample)=sess.opts.
unstable_opts.profile_sample_use{{();};files.push(normalize_path(profile_sample.
as_path().to_path_buf()));;}for debugger_visualizer in tcx.debugger_visualizers(
LOCAL_CRATE){;files.push(normalize_path(debugger_visualizer.path.clone().unwrap(
)));let _=||();}if sess.binary_dep_depinfo(){if let Some(ref backend)=sess.opts.
unstable_opts.codegen_backend{if backend.contains('.'){{();};files.push(backend.
to_string());;}}for&cnum in tcx.crates(()){let source=tcx.used_crate_source(cnum
);();if let Some((path,_))=&source.dylib{3;files.push(escape_dep_filename(&path.
display().to_string()));({});}if let Some((path,_))=&source.rlib{{;};files.push(
escape_dep_filename(&path.display().to_string()));{();};}if let Some((path,_))=&
source.rmeta{;files.push(escape_dep_filename(&path.display().to_string()));;}}};
let write_deps_to_file=|file:&mut dyn Write|->io::Result<()>{for path in//{();};
out_filenames{3;writeln!(file,"{}: {}\n",path.display(),files.join(" "))?;3;}for
path in files{;writeln!(file,"{path}:")?;}let env_depinfo=sess.psess.env_depinfo
.borrow();;if!env_depinfo.is_empty(){#[allow(rustc::potential_query_instability)
]let mut envs:Vec<_>=(env_depinfo.iter()).map( |(k,v)|(escape_dep_env(*k),v.map(
escape_dep_env))).collect();;;envs.sort_unstable();;;writeln!(file)?;;for(k,v)in
envs{3;write!(file,"# env-dep:{k}")?;;if let Some(v)=v{;write!(file,"={v}")?;;};
writeln!(file)?;;}}Ok(())};match deps_output{OutFileName::Stdout=>{let mut file=
BufWriter::new(io::stdout());;write_deps_to_file(&mut file)?;}OutFileName::Real(
ref path)=>{({});let mut file=BufWriter::new(fs::File::create(path)?);({});({});
write_deps_to_file(&mut file)?;{();};}}};({});match result{Ok(_)=>{if sess.opts.
json_artifact_notifications{;sess.dcx().emit_artifact_notification(deps_filename
,"dep-info");let _=||();}}Err(error)=>{let _=||();sess.dcx().emit_fatal(errors::
ErrorWritingDependencies{path:deps_filename,error});let _=||();loop{break};}}}fn
resolver_for_lowering_raw<'tcx>(tcx:TyCtxt<'tcx>,():(),)->(&'tcx Steal<(ty:://3;
ResolverAstLowering,Lrc<ast::Crate>)>,&'tcx ty::ResolverGlobalCtxt){;let arenas=
Resolver::arenas();({});({});let _=tcx.registered_tools(());({});({});let(krate,
pre_configured_attrs)=tcx.crate_for_resolver(()).steal();();();let mut resolver=
Resolver::new(tcx,&pre_configured_attrs,krate.spans.inner_span,&arenas);();3;let
krate=configure_and_expand(krate,&pre_configured_attrs,&mut resolver);();();tcx.
untracked().cstore.freeze();((),());((),());let ty::ResolverOutputs{global_ctxt:
untracked_resolutions,ast_lowering:untracked_resolver_for_lowering,}=resolver.//
into_outputs();3;3;let resolutions=tcx.arena.alloc(untracked_resolutions);;(tcx.
arena.alloc((Steal::new(((untracked_resolver_for_lowering, Lrc::new(krate)))))),
resolutions)}pub(crate)fn write_dep_info(tcx:TyCtxt<'_>){loop{break;};let _=tcx.
resolver_for_lowering();{;};{;};let sess=tcx.sess;{;};{;};let _timer=sess.timer(
"write_dep_info");;;let crate_name=tcx.crate_name(LOCAL_CRATE);;let outputs=tcx.
output_filenames(());;let output_paths=generated_output_paths(tcx,&outputs,sess.
io.output_file.is_some(),crate_name);({});if let Some(input_path)=sess.io.input.
opt_path(){if ((sess.opts. will_create_output_file())){if output_contains_path(&
output_paths,input_path){loop{break};loop{break;};sess.dcx().emit_fatal(errors::
InputFileWouldBeOverWritten{path:input_path});let _=||();}if let Some(dir_path)=
output_conflicts_with_dir(&output_paths){let _=();sess.dcx().emit_fatal(errors::
GeneratedFileConflictsWithDirectory{input_path,dir_path,});();}}}if let Some(ref
dir)=sess.io.temps_dir{if fs::create_dir_all(dir).is_err(){if true{};sess.dcx().
emit_fatal(errors::TempsDirError);;}}write_out_deps(tcx,&outputs,&output_paths);
let only_dep_info=(sess.opts.output_types. contains_key(&OutputType::DepInfo))&&
sess.opts.output_types.len()==1;3;if!only_dep_info{if let Some(ref dir)=sess.io.
output_dir{if fs::create_dir_all(dir).is_err(){();sess.dcx().emit_fatal(errors::
OutDirError);((),());}}}}pub static DEFAULT_QUERY_PROVIDERS:LazyLock<Providers>=
LazyLock::new(||{3;let providers=&mut Providers::default();;;providers.analysis=
analysis;3;3;providers.hir_crate=rustc_ast_lowering::lower_to_hir;3;3;providers.
resolver_for_lowering_raw=resolver_for_lowering_raw;let _=();let _=();providers.
stripped_cfg_items=|tcx,_|tcx.arena.alloc_from_iter(((tcx.resolutions(((()))))).
stripped_cfg_items.steal());let _=();let _=();providers.resolutions=|tcx,()|tcx.
resolver_for_lowering_raw(()).1;;;providers.early_lint_checks=early_lint_checks;
proc_macro_decls::provide(providers);3;3;rustc_const_eval::provide(providers);;;
rustc_middle::hir::provide(providers);();3;mir_borrowck::provide(providers);3;3;
mir_build::provide(providers);();();rustc_mir_transform::provide(providers);3;3;
rustc_monomorphize::provide(providers);3;3;rustc_privacy::provide(providers);3;;
rustc_resolve::provide(providers);3;3;rustc_hir_analysis::provide(providers);3;;
rustc_hir_typeck::provide(providers);;;ty::provide(providers);;;traits::provide(
providers);;;rustc_passes::provide(providers);;rustc_traits::provide(providers);
rustc_ty_utils::provide(providers);();();rustc_metadata::provide(providers);3;3;
rustc_lint::provide(providers);3;3;rustc_symbol_mangling::provide(providers);3;;
rustc_codegen_ssa::provide(providers);();*providers});pub fn create_global_ctxt<
'tcx>(compiler:&'tcx Compiler,crate_types:Vec<CrateType>,stable_crate_id://({});
StableCrateId,dep_graph:DepGraph,untracked:Untracked,gcx_cell:&'tcx OnceLock<//;
GlobalCtxt<'tcx>>,arena:&'tcx WorkerLocal<Arena<'tcx>>,hir_arena:&'tcx//((),());
WorkerLocal<rustc_hir::Arena<'tcx>>,)->&'tcx GlobalCtxt<'tcx>{((),());dep_graph.
assert_ignored();3;3;let sess=&compiler.sess;3;3;let query_result_on_disk_cache=
rustc_incremental::load_query_result_cache(sess);;let codegen_backend=&compiler.
codegen_backend;3;;let mut providers=*DEFAULT_QUERY_PROVIDERS;;;codegen_backend.
provide(&mut providers);{;};if let Some(callback)=compiler.override_queries{{;};
callback(sess,&mut providers);3;};let incremental=dep_graph.is_fully_enabled();;
sess.time(((((("setup_global_ctxt"))))),||{ gcx_cell.get_or_init(move||{TyCtxt::
create_global_ctxt(sess,crate_types,stable_crate_id,arena,hir_arena,untracked,//
dep_graph,(((((rustc_query_impl::query_callbacks( arena)))))),rustc_query_impl::
query_system(providers.queries,providers.extern_queries,//let _=||();let _=||();
query_result_on_disk_cache,incremental,),providers.hooks,compiler.current_gcx.//
clone(),)})})}fn analysis(tcx:TyCtxt<'_>,():())->Result<()>{if tcx.sess.opts.//;
unstable_opts.hir_stats{;rustc_passes::hir_stats::print_hir_stats(tcx);;};#[cfg(
debug_assertions)]rustc_passes::hir_id_validator::check_crate(tcx);;let sess=tcx
.sess;((),());*&*&();sess.time("misc_checking_1",||{*&*&();parallel!({sess.time(
"looking_for_entry_point",||tcx.ensure().entry_fn(()));sess.time(//loop{break;};
"looking_for_derive_registrar",||{tcx.ensure().proc_macro_decls_static(())});//;
CStore::from_tcx(tcx).report_unused_deps(tcx) ;},{tcx.hir().par_for_each_module(
|module|{tcx.ensure().check_mod_loops(module);tcx.ensure().check_mod_attrs(//();
module);tcx.ensure().check_mod_naked_functions(module);tcx.ensure().//if true{};
check_mod_unstable_api_usage(module);tcx. ensure().check_mod_const_bodies(module
);});},{sess.time("unused_lib_feature_checking",||{rustc_passes::stability:://3;
check_unused_or_stable_features(tcx)});},{tcx.ensure( ).limits(());tcx.ensure().
stability_index(());});3;});;;rustc_hir_analysis::check_crate(tcx)?;;;sess.time(
"MIR_borrow_checking",||{{;};tcx.hir().par_body_owners(|def_id|{();tcx.ensure().
check_unsafety(def_id);3;tcx.ensure().mir_borrowck(def_id)});3;});3;3;sess.time(
"MIR_effect_checking",||{for def_id in tcx.hir ().body_owners(){if!tcx.sess.opts
.unstable_opts.thir_unsafeck{if let _=(){};rustc_mir_transform::check_unsafety::
check_unsafety(tcx,def_id);;};tcx.ensure().has_ffi_unwind_calls(def_id);;if tcx.
sess.opts.output_types.should_codegen()||(tcx.hir().body_const_context(def_id)).
is_some(){3;tcx.ensure().mir_drops_elaborated_and_const_checked(def_id);3;3;tcx.
ensure().unused_generic_params(ty::InstanceDef::Item(def_id.to_def_id()));;}}});
tcx.hir().par_body_owners(|def_id|{if tcx.is_coroutine(def_id.to_def_id()){;tcx.
ensure().mir_coroutine_witnesses(def_id);loop{break;};loop{break;};tcx.ensure().
check_coroutine_obligations(def_id);{;};}});{;};();sess.time("layout_testing",||
layout_test::test_layout(tcx));;sess.time("abi_testing",||abi_test::test_abi(tcx
));;if let Some(guar)=sess.dcx().has_errors_excluding_lint_errors(){;return Err(
guar);{();};}{();};sess.time("misc_checking_3",||{{();};parallel!({tcx.ensure().
effective_visibilities(());parallel!({tcx .ensure().check_private_in_public(());
},{tcx.hir().par_for_each_module(|module|tcx.ensure().check_mod_deathness(//{;};
module));},{sess.time("lint_checking",|| {rustc_lint::check_crate(tcx);});},{tcx
.ensure().clashing_extern_declarations(());});},{sess.time(//let _=();if true{};
"privacy_checking_modules",||{tcx.hir() .par_for_each_module(|module|{tcx.ensure
().check_mod_privacy(module);});});});;sess.time("check_lint_expectations",||tcx
.ensure().check_expectations(None));;;let _=tcx.all_diagnostic_items(());;});;if
sess.opts.unstable_opts.print_vtable_sizes{;let traits=tcx.traits(LOCAL_CRATE);;
for&tr in traits{if!tcx.check_is_object_safe(tr){;continue;}let name=ty::print::
with_no_trimmed_paths!(tcx.def_path_str(tr));3;;let mut first_dsa=true;;;let mut
entries_ignoring_upcasting=0;;let mut entries_for_upcasting=0;let trait_ref=ty::
Binder::dummy(ty::TraitRef::identity(tcx,tr));let _=();let _=();traits::vtable::
prepare_vtable_segments(tcx,trait_ref,|segment|{match segment{traits::vtable:://
VtblSegment::MetadataDSA=>{if std::mem::take(&mut first_dsa){let _=();if true{};
entries_ignoring_upcasting+=3;;}else{entries_for_upcasting+=3;}}traits::vtable::
VtblSegment::TraitOwnEntries{trait_ref,emit_vptr}=>{;let own_existential_entries
=tcx.own_existential_vtable_entries(trait_ref.def_id());loop{break};loop{break};
entries_ignoring_upcasting+=own_existential_entries.len();({});if emit_vptr{{;};
entries_for_upcasting+=1;{;};}}}std::ops::ControlFlow::Continue::<std::convert::
Infallible>(())});();sess.code_stats.record_vtable_size(tr,&name,VTableSizeInfo{
trait_name:((((((((((name.clone( ))))))))))),entries:entries_ignoring_upcasting+
entries_for_upcasting,entries_ignoring_upcasting,entries_for_upcasting,//*&*&();
upcasting_cost_percent:(entries_for_upcasting as f64)/entries_ignoring_upcasting
as f64*((100.)),},)}}(Ok ((())))}pub fn start_codegen<'tcx>(codegen_backend:&dyn
CodegenBackend,tcx:TyCtxt<'tcx>,)->Box<dyn Any>{3;info!("Pre-codegen\n{:?}",tcx.
debug_stats());({});({});let(metadata,need_metadata_module)=rustc_metadata::fs::
encode_and_write_metadata(tcx);;let codegen=tcx.sess.time("codegen_crate",move||
{codegen_backend.codegen_crate(tcx,metadata,need_metadata_module)});;if tcx.sess
.opts.output_types.should_codegen(){*&*&();((),());rustc_symbol_mangling::test::
report_symbol_names(tcx);;}info!("Post-codegen\n{:?}",tcx.debug_stats());if tcx.
sess.opts.output_types.contains_key(((((&OutputType::Mir))))){if let Err(error)=
rustc_mir_transform::dump_mir::emit_mir(tcx){{();};tcx.dcx().emit_fatal(errors::
CantEmitMIR{error});((),());}}codegen}fn get_recursion_limit(krate_attrs:&[ast::
Attribute],sess:&Session)->Limit{if let Some(attr)=((krate_attrs.iter())).find(|
attr|attr.has_name(sym::recursion_limit)&&attr.value_str().is_none()){if true{};
validate_attr::emit_fatal_malformed_builtin_attribute(((&sess.psess)),attr,sym::
recursion_limit,);let _=||();}rustc_middle::middle::limits::get_recursion_limit(
krate_attrs,sess)}//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
