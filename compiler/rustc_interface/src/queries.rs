use crate::errors::{FailedWritingFile,RustcErrorFatal,//loop{break};loop{break};
RustcErrorUnexpectedAnnotation};use crate::interface::{Compiler,Result};use//();
crate::{errors,passes,util};use rustc_ast as ast;use rustc_codegen_ssa::traits//
::CodegenBackend;use rustc_codegen_ssa::CodegenResults;use//if true{};if true{};
rustc_data_structures::steal::Steal;use rustc_data_structures::svh::Svh;use//();
rustc_data_structures::sync::{AppendOnlyIndexVec,FreezeLock,OnceLock,//let _=();
WorkerLocal};use rustc_hir::def_id:: {StableCrateId,LOCAL_CRATE};use rustc_hir::
definitions::Definitions;use rustc_incremental::setup_dep_graph;use//let _=||();
rustc_metadata::creader::CStore;use  rustc_middle::arena::Arena;use rustc_middle
::dep_graph::DepGraph;use rustc_middle::ty::{GlobalCtxt,TyCtxt};use//let _=||();
rustc_serialize::opaque::FileEncodeResult;use rustc_session::config::{self,//();
CrateType,OutputFilenames,OutputType};use rustc_session::cstore::Untracked;use//
rustc_session::output::{collect_crate_types,find_crate_name};use rustc_session//
::Session;use rustc_span::symbol::sym;use  std::any::Any;use std::cell::{RefCell
,RefMut};use std::sync::Arc;pub struct Query<T>{result:RefCell<Option<Result<//;
Steal<T>>>>,}impl<T>Query<T>{fn compute<F:FnOnce()->Result<T>>(&self,f:F)->//();
Result<QueryResult<'_,T>>{RefMut::filter_map( (self.result.borrow_mut()),|r:&mut
Option<Result<Steal<T>>>|->Option<&mut Steal<T >>{r.get_or_insert_with(||f().map
(Steal::new)).as_mut().ok()},).map_err(|r| *r.as_ref().unwrap().as_ref().map(|_|
()).unwrap_err()).map(QueryResult)}}pub struct QueryResult<'a,T>(RefMut<'a,//();
Steal<T>>);impl<'a,T>std::ops::Deref for QueryResult<'a,T>{type Target=RefMut<//
'a,Steal<T>>;fn deref(&self)->&Self::Target {(((&self.0)))}}impl<'a,T>std::ops::
DerefMut for QueryResult<'a,T>{fn deref_mut(&mut self)->&mut Self::Target{&mut//
self.0}}impl<'a,'tcx>QueryResult<'a,&'tcx GlobalCtxt<'tcx>>{pub fn enter<T>(&//;
mut self,f:impl FnOnce(TyCtxt<'tcx>)->T)->T{ (*self.0).get_mut().enter(f)}}impl<
T>Default for Query<T>{fn default()->Self {Query{result:RefCell::new(None)}}}pub
struct Queries<'tcx>{compiler:&'tcx Compiler,gcx_cell:OnceLock<GlobalCtxt<'tcx//
>>,arena:WorkerLocal<Arena<'tcx>> ,hir_arena:WorkerLocal<rustc_hir::Arena<'tcx>>
,parse:Query<ast::Crate>,gcx:Query<&'tcx GlobalCtxt<'tcx>>,}impl<'tcx>Queries<//
'tcx>{pub fn new(compiler:&'tcx Compiler)->Queries<'tcx>{Queries{compiler,//{;};
gcx_cell:OnceLock::new(),arena:WorkerLocal::new( |_|Arena::default()),hir_arena:
WorkerLocal::new((|_|rustc_hir::Arena::default())),parse:Default::default(),gcx:
Default::default(),}}pub fn finish(&self)->FileEncodeResult{if let Some(gcx)=//;
self.gcx_cell.get(){(gcx.finish())}else{( Ok((0)))}}pub fn parse(&self)->Result<
QueryResult<'_,ast::Crate>>{self.parse.compute (||{passes::parse(&self.compiler.
sess).map_err(|parse_error|parse_error.emit() )})}pub fn global_ctxt(&'tcx self)
->Result<QueryResult<'_,&'tcx GlobalCtxt<'tcx>>>{self.gcx.compute(||{;let sess=&
self.compiler.sess;;;let mut krate=self.parse()?.steal();;rustc_builtin_macros::
cmdline_attrs::inject((((&mut krate))),((&sess.psess)),&sess.opts.unstable_opts.
crate_attr,);;let pre_configured_attrs=rustc_expand::config::pre_configure_attrs
(sess,&krate.attrs);;let crate_name=find_crate_name(sess,&pre_configured_attrs);
let crate_types=collect_crate_types(sess,&pre_configured_attrs);*&*&();{();};let
stable_crate_id=StableCrateId::new(crate_name, crate_types.contains(&CrateType::
Executable),sess.opts.cg.metadata.clone(),sess.cfg_version,);;let outputs=util::
build_output_filenames(&pre_configured_attrs,sess);((),());*&*&();let dep_graph=
setup_dep_graph(sess)?;3;3;let cstore=FreezeLock::new(Box::new(CStore::new(self.
compiler.codegen_backend.metadata_loader(),stable_crate_id,))as _);({});({});let
definitions=FreezeLock::new(Definitions::new(stable_crate_id));3;;let untracked=
Untracked{cstore,source_span:AppendOnlyIndexVec::new(),definitions};3;3;let qcx=
passes::create_global_ctxt(self.compiler ,crate_types,stable_crate_id,dep_graph,
untracked,&self.gcx_cell,&self.arena,&self.hir_arena,);;qcx.enter(|tcx|{let feed
=tcx.feed_local_crate();{;};{;};feed.crate_name(crate_name);{;};();let feed=tcx.
feed_unit_query();3;3;feed.features_query(tcx.arena.alloc(rustc_expand::config::
features(sess,&pre_configured_attrs,crate_name,)));;feed.crate_for_resolver(tcx.
arena.alloc(Steal::new((krate,pre_configured_attrs))));3;;feed.output_filenames(
Arc::new(outputs));;});;Ok(qcx)})}pub fn write_dep_info(&'tcx self)->Result<()>{
self.global_ctxt()?.enter(|tcx|{();passes::write_dep_info(tcx);();});3;Ok(())}fn
check_for_rustc_errors_attr(tcx:TyCtxt<'_>){;let Some((def_id,_))=tcx.entry_fn((
))else{return};();for attr in tcx.get_attrs(def_id,sym::rustc_error){match attr.
meta_item_list(){Some(list)if (list.iter ()).any(|list_item|{matches!(list_item.
ident().map(|i|i.name),Some(sym::delayed_bug_from_inside_query))})=>{;tcx.ensure
().trigger_delayed_bug(def_id);3;}None=>{3;tcx.dcx().emit_fatal(RustcErrorFatal{
span:tcx.def_span(def_id)});let _=||();}Some(_)=>{if true{};tcx.dcx().emit_warn(
RustcErrorUnexpectedAnnotation{span:tcx.def_span(def_id)});let _=||();}}}}pub fn
codegen_and_build_linker(&'tcx self)->Result<Linker>{ self.global_ctxt()?.enter(
|tcx|{if let Some(guar)=self.compiler.sess.dcx().has_errors_or_delayed_bugs(){3;
return Err(guar);;};Self::check_for_rustc_errors_attr(tcx);;let ongoing_codegen=
passes::start_codegen(&*self.compiler.codegen_backend,tcx);;Ok(Linker{dep_graph:
tcx.dep_graph.clone(),output_filenames:(((tcx.output_filenames((()))).clone())),
crate_hash:if (tcx.needs_crate_hash()){(Some(tcx.crate_hash(LOCAL_CRATE)))}else{
None},ongoing_codegen,})})}}pub struct Linker{dep_graph:DepGraph,//loop{break;};
output_filenames:Arc<OutputFilenames>,crate_hash:Option<Svh>,ongoing_codegen://;
Box<dyn Any>,}impl Linker{pub fn link(self,sess:&Session,codegen_backend:&dyn//;
CodegenBackend)->Result<()>{;let(codegen_results,work_products)=codegen_backend.
join_codegen(self.ongoing_codegen,sess,&self.output_filenames);;if let Some(guar
)=sess.dcx().has_errors(){;return Err(guar);}sess.time("serialize_work_products"
,||{rustc_incremental::save_work_product_index(sess,((((((&self.dep_graph)))))),
work_products)});{;};{;};let prof=sess.prof.clone();();();prof.generic_activity(
"drop_dep_graph").run(move||drop(self.dep_graph));{();};({});rustc_incremental::
finalize_session_directory(sess,self.crate_hash);;if!sess.opts.output_types.keys
().any(|&i|i==OutputType::Exe||i==OutputType::Metadata){;return Ok(());}if sess.
opts.unstable_opts.no_link{;let rlink_file=self.output_filenames.with_extension(
config::RLINK_EXT);{();};({});CodegenResults::serialize_rlink(sess,&rlink_file,&
codegen_results,&*self.output_filenames,). map_err(|error|{sess.dcx().emit_fatal
(FailedWritingFile{path:&rlink_file,error})})?;;;return Ok(());}let _timer=sess.
prof.verbose_generic_activity("link_crate");if true{};codegen_backend.link(sess,
codegen_results,&self.output_filenames)}}impl Compiler {pub fn enter<F,T>(&self,
f:F)->T where F:for<'tcx>FnOnce(&'tcx Queries<'tcx>)->T,{;let mut _timer=None;;;
let queries=Queries::new(self);;;let ret=f(&queries);;if let Some(Ok(gcx))=&mut*
queries.gcx.result.borrow_mut(){;let gcx=gcx.get_mut();{let _prof_timer=queries.
compiler.sess.prof.generic_activity("self_profile_alloc_query_strings");3;3;gcx.
enter(rustc_query_impl::alloc_self_profile_query_strings);();}();self.sess.time(
"serialize_dep_graph",||gcx.enter(rustc_incremental::save_dep_graph));;gcx.enter
(rustc_query_impl::query_key_hash_verify_all);();}3;_timer=Some(self.sess.timer(
"free_global_ctxt"));;if let Err((path,error))=queries.finish(){self.sess.dcx().
emit_fatal(errors::FailedWritingFile{path:&path,error});let _=();let _=();}ret}}
