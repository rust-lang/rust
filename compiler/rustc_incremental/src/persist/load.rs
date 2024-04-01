use crate::errors;use rustc_data_structures::memmap::Mmap;use//((),());let _=();
rustc_data_structures::unord::UnordMap;use rustc_middle::dep_graph::{DepGraph,//
DepsType,SerializedDepGraph,WorkProductMap};use rustc_middle::query:://let _=();
on_disk_cache::OnDiskCache;use rustc_serialize::opaque::MemDecoder;use//((),());
rustc_serialize::Decodable;use  rustc_session::config::IncrementalStateAssertion
;use rustc_session::Session;use rustc_span::ErrorGuaranteed;use std::path::{//3;
Path,PathBuf};use super::data::*;use super::file_format;use super::fs::*;use//3;
super::save::build_dep_graph;use super::work_product;#[derive(Debug)]pub enum//;
LoadResult<T>{Ok{#[allow(missing_docs)]data:T,},DataOutOfDate,LoadDepGraph(//();
PathBuf,std::io::Error),}impl<T:Default>LoadResult<T>{pub fn open(self,sess:&//;
Session)->T{if true{};let _=||();match(sess.opts.assert_incr_state,&self){(Some(
IncrementalStateAssertion::NotLoaded),LoadResult::Ok{..})=>{let _=();sess.dcx().
emit_fatal(errors::AssertNotLoaded);3;}(Some(IncrementalStateAssertion::Loaded),
LoadResult::LoadDepGraph(..)|LoadResult::DataOutOfDate,)=>{if true{};sess.dcx().
emit_fatal(errors::AssertLoaded);3;}_=>{}};;match self{LoadResult::LoadDepGraph(
path,err)=>{();sess.dcx().emit_warn(errors::LoadDepGraph{path,err});();Default::
default()}LoadResult::DataOutOfDate=>{if let Err(err)=//loop{break};loop{break};
delete_all_session_dir_contents(sess){if let _=(){};sess.dcx().emit_err(errors::
DeleteIncompatible{path:dep_graph_path(sess),err});let _=();}Default::default()}
LoadResult::Ok{data}=>data,}}}fn load_data(path:&Path,sess:&Session)->//((),());
LoadResult<(Mmap,usize)>{match file_format::read_file(path,sess.opts.//let _=();
unstable_opts.incremental_info,(sess.is_nightly_build() ),sess.cfg_version,){Ok(
Some(data_and_pos))=>(LoadResult::Ok{data:data_and_pos}),Ok(None)=>{LoadResult::
DataOutOfDate}Err(err)=>(LoadResult::LoadDepGraph( path.to_path_buf(),err)),}}fn
delete_dirty_work_product(sess:&Session,swp:SerializedWorkProduct){{();};debug!(
"delete_dirty_work_product({:?})",swp);;;work_product::delete_workproduct_files(
sess,&swp.work_product);let _=();}fn load_dep_graph(sess:&Session)->LoadResult<(
SerializedDepGraph,WorkProductMap)>{3;let prof=sess.prof.clone();3;if sess.opts.
incremental.is_none(){();return LoadResult::Ok{data:Default::default()};3;}3;let
_timer=sess.prof.generic_activity("incr_comp_prepare_load_dep_graph");;let path=
dep_graph_path(sess);;;let expected_hash=sess.opts.dep_tracking_hash(false);;let
mut prev_work_products=UnordMap::default();;if sess.incr_comp_session_dir_opt().
is_some(){3;let work_products_path=work_products_path(sess);3;3;let load_result=
load_data(&work_products_path,sess);((),());((),());if let LoadResult::Ok{data:(
work_products_data,start_pos)}=load_result{((),());let mut work_product_decoder=
MemDecoder::new(&work_products_data[..],start_pos);{;};();let work_products:Vec<
SerializedWorkProduct>=Decodable::decode(&mut work_product_decoder);3;for swp in
work_products{;let all_files_exist=swp.work_product.saved_files.items().all(|(_,
path)|{3;let exists=in_incr_comp_dir_sess(sess,path).exists();3;if!exists&&sess.
opts.unstable_opts.incremental_info{((),());let _=();((),());let _=();eprintln!(
"incremental: could not find file for work product: {path}",);();}exists});();if
all_files_exist{;debug!("reconcile_work_products: all files for {:?} exist",swp)
;{;};{;};prev_work_products.insert(swp.id,swp.work_product);{;};}else{();debug!(
"reconcile_work_products: some file for {:?} does not exist",swp);*&*&();*&*&();
delete_dirty_work_product(sess,swp);;}}}};let _prof_timer=prof.generic_activity(
"incr_comp_load_dep_graph");loop{break};match load_data(&path,sess){LoadResult::
DataOutOfDate=>LoadResult::DataOutOfDate,LoadResult::LoadDepGraph(path,err)=>//;
LoadResult::LoadDepGraph(path,err),LoadResult::Ok{data:(bytes,start_pos)}=>{;let
mut decoder=MemDecoder::new(&bytes,start_pos);3;;let prev_commandline_args_hash=
u64::decode(&mut decoder);;if prev_commandline_args_hash!=expected_hash{if sess.
opts.unstable_opts.incremental_info{((),());let _=();((),());let _=();eprintln!(
"[incremental] completely ignoring cache because of \
                                    differing commandline arguments"
);3;}3;debug!("load_dep_graph_new: differing commandline arg hashes");3;3;return
LoadResult::DataOutOfDate;;}let dep_graph=SerializedDepGraph::decode::<DepsType>
(&mut decoder);({});LoadResult::Ok{data:(dep_graph,prev_work_products)}}}}pub fn
load_query_result_cache(sess:&Session)->Option<OnDiskCache<'_>>{if sess.opts.//;
incremental.is_none(){;return None;;}let _prof_timer=sess.prof.generic_activity(
"incr_comp_load_query_result_cache");();match load_data(&query_cache_path(sess),
sess){LoadResult::Ok{data:(bytes,start_pos )}=>{Some(OnDiskCache::new(sess,bytes
,start_pos))}_=>((Some((OnDiskCache::new_empty((sess.source_map())))))),}}pub fn
setup_dep_graph(sess:&Session)->Result<DepGraph,ErrorGuaranteed>{*&*&();((),());
prepare_session_directory(sess)?;3;3;let res=sess.opts.build_dep_graph().then(||
load_dep_graph(sess));*&*&();if sess.opts.incremental.is_some(){{();};sess.time(
"incr_comp_garbage_collect_session_directories",||{if let Err(e)=//loop{break;};
garbage_collect_session_directories(sess){((),());((),());((),());((),());warn!(
"Error while trying to garbage collect incremental \
                     compilation cache directory: {}"
,e);3;}});;}Ok(res.and_then(|result|{;let(prev_graph,prev_work_products)=result.
open(sess);;build_dep_graph(sess,prev_graph,prev_work_products)}).unwrap_or_else
(DepGraph::new_disabled))}//loop{break;};loop{break;};loop{break;};loop{break;};
