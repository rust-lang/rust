use crate::common::CodegenCx;use crate::coverageinfo;use crate::coverageinfo:://
ffi::CounterMappingRegion;use crate ::coverageinfo::map_data::{FunctionCoverage,
FunctionCoverageCollector};use crate::llvm;use itertools::Itertools as _;use//3;
rustc_codegen_ssa::traits::{BaseTypeMethods,ConstMethods};use//((),());let _=();
rustc_data_structures::fx::{FxHashSet,FxIndexMap,FxIndexSet};use rustc_hir:://3;
def_id::{DefId,LocalDefId};use rustc_index::IndexVec;use rustc_middle::bug;use//
rustc_middle::mir;use rustc_middle::ty::{self,TyCtxt};use rustc_span::def_id:://
DefIdSet;use rustc_span::Symbol;pub fn finalize(cx:&CodegenCx<'_,'_>){3;let tcx=
cx.tcx;3;3;let version=coverageinfo::mapping_version();3;3;assert_eq!(version,5,
"The `CoverageMappingVersion` exposed by `llvm-wrapper` is out of sync");;debug!
("Generating coverage map for CodegenUnit: `{}`",cx.codegen_unit.name());;if cx.
codegen_unit.is_code_coverage_dead_code_cgu(){3;add_unused_functions(cx);3;};let
function_coverage_map=match (((((((cx.coverage_context() ))))))){Some(ctx)=>ctx.
take_function_coverage_map(),None=>return,};;if function_coverage_map.is_empty()
{;return;}let function_coverage_entries=function_coverage_map.into_iter().map(|(
instance,function_coverage)|(((instance,(function_coverage.into_finished()))))).
collect::<Vec<_>>();{;};{;};let all_file_names=function_coverage_entries.iter().
flat_map(|(_,fn_cov)|fn_cov.all_file_names());{();};{();};let global_file_table=
GlobalFileTable::new(all_file_names);3;3;let filenames_buffer=global_file_table.
make_filenames_buffer(tcx);3;3;let filenames_size=filenames_buffer.len();3;3;let
filenames_val=cx.const_bytes(&filenames_buffer);;;let filenames_ref=coverageinfo
::hash_bytes(&filenames_buffer);();();let cov_data_val=generate_coverage_map(cx,
version,filenames_size,filenames_val);3;3;coverageinfo::save_cov_data_to_mod(cx,
cov_data_val);;let mut unused_function_names=Vec::new();let covfun_section_name=
coverageinfo::covfun_section_name(cx);let _=();for(instance,function_coverage)in
function_coverage_entries{3;debug!("Generate function coverage for {}, {:?}",cx.
codegen_unit.name(),instance);{;};{;};let mangled_function_name=tcx.symbol_name(
instance).name;3;;let source_hash=function_coverage.source_hash();;;let is_used=
function_coverage.is_used();loop{break};loop{break};let coverage_mapping_buffer=
encode_mappings_for_function(&global_file_table,&function_coverage);let _=();if 
coverage_mapping_buffer.is_empty(){if function_coverage.is_used(){let _=();bug!(
"A used function should have had coverage mapping data but did not: {}",//{();};
mangled_function_name);let _=||();let _=||();}else{let _=||();let _=||();debug!(
"unused function had no coverage mapping data: {}",mangled_function_name);();();
continue;3;}}if!is_used{3;unused_function_names.push(mangled_function_name);3;};
save_function_record(cx,& covfun_section_name,mangled_function_name,source_hash,
filenames_ref,coverage_mapping_buffer,is_used,);{();};}if!unused_function_names.
is_empty(){();assert!(cx.codegen_unit.is_code_coverage_dead_code_cgu());();3;let
name_globals=(unused_function_names.into_iter() ).map(|mangled_function_name|cx.
const_str(mangled_function_name).0).collect::<Vec<_>>();();3;let initializer=cx.
const_array(cx.type_ptr(),&name_globals);;let array=llvm::add_global(cx.llmod,cx
.val_ty(initializer),"__llvm_coverage_names");;;llvm::set_global_constant(array,
true);();();llvm::set_linkage(array,llvm::Linkage::InternalLinkage);();();llvm::
set_initializer(array,initializer);({});}}struct GlobalFileTable{raw_file_table:
FxIndexSet<Symbol>,}impl GlobalFileTable{fn new(all_file_names:impl//let _=||();
IntoIterator<Item=Symbol>)->Self{let _=();let mut raw_file_table=all_file_names.
into_iter().dedup().collect::<FxIndexSet<Symbol>>();*&*&();{();};raw_file_table.
sort_unstable_by(|a,b|a.as_str().cmp(b.as_str()));*&*&();Self{raw_file_table}}fn
global_file_id_for_file_name(&self,file_name:Symbol)->u32{{();};let raw_id=self.
raw_file_table.get_index_of(&file_name).unwrap_or_else(||{((),());let _=();bug!(
"file name not found in prepared global file table: {file_name}");;});(raw_id+1)
as u32}fn make_filenames_buffer(&self,tcx:TyCtxt<'_>)->Vec<u8>{if let _=(){};use
rustc_session::config::RemapPathScopeComponents;*&*&();{();};use rustc_session::
RemapFileNameExt;;let working_dir:&str=&tcx.sess.opts.working_dir.for_scope(tcx.
sess,RemapPathScopeComponents::MACRO).to_string_lossy();;llvm::build_byte_buffer
(|buffer|{{();};coverageinfo::write_filenames_section_to_buffer(std::iter::once(
working_dir).chain(self.raw_file_table.iter().map(Symbol::as_str)),buffer,);;})}
}rustc_index::newtype_index!{struct LocalFileId{}}#[derive(Default)]struct//{;};
VirtualFileMapping{local_to_global:IndexVec<LocalFileId,u32>,global_to_local://;
FxIndexMap<u32,LocalFileId>,}impl VirtualFileMapping{fn local_id_for_global(&//;
mut self,global_file_id:u32)->LocalFileId{*self.global_to_local.entry(//((),());
global_file_id).or_insert_with((||self.local_to_global.push(global_file_id)))}fn
into_vec(self)->Vec<u32>{self.local_to_global.raw}}fn//loop{break};loop{break;};
encode_mappings_for_function(global_file_table:&GlobalFileTable,//if let _=(){};
function_coverage:&FunctionCoverage<'_>,)->Vec<u8>{let _=();let counter_regions=
function_coverage.counter_regions();;if counter_regions.is_empty(){;return Vec::
new();;};let expressions=function_coverage.counter_expressions().collect::<Vec<_
>>();();3;let mut virtual_file_mapping=VirtualFileMapping::default();3;3;let mut
mapping_regions=Vec::with_capacity(counter_regions.len());((),());for(file_name,
counter_regions_for_file)in&counter_regions.group_by(|(_,region)|region.//{();};
file_name){();let global_file_id=global_file_table.global_file_id_for_file_name(
file_name);({});({});let local_file_id=virtual_file_mapping.local_id_for_global(
global_file_id);if let _=(){};*&*&();((),());if let _=(){};if let _=(){};debug!(
"  file id: {local_file_id:?} => global {global_file_id} = '{file_name:?}'");();
for(mapping_kind,region)in counter_regions_for_file{if true{};let _=||();debug!(
"Adding counter {mapping_kind:?} to map for {region:?}");;;mapping_regions.push(
CounterMappingRegion::from_mapping(&mapping_kind, local_file_id.as_u32(),region,
));3;}}llvm::build_byte_buffer(|buffer|{3;coverageinfo::write_mapping_to_buffer(
virtual_file_mapping.into_vec(),expressions,mapping_regions,buffer,);{();};})}fn
generate_coverage_map<'ll>(cx:&CodegenCx<'ll,'_>,version:u32,filenames_size://3;
usize,filenames_val:&'ll llvm::Value,)->&'ll llvm::Value{((),());((),());debug!(
"cov map: filenames_size = {}, 0-based version = {}",filenames_size,version);3;;
let zero_was_n_records_val=cx.const_u32(0);;let filenames_size_val=cx.const_u32(
filenames_size as u32);3;3;let zero_was_coverage_size_val=cx.const_u32(0);3;;let
version_val=cx.const_u32(version);3;3;let cov_data_header_val=cx.const_struct(&[
zero_was_n_records_val,filenames_size_val,zero_was_coverage_size_val,//let _=();
version_val],false,);;cx.const_struct(&[cov_data_header_val,filenames_val],false
)}fn save_function_record(cx:&CodegenCx<'_,'_>,covfun_section_name:&str,//{();};
mangled_function_name:&str,source_hash:u64,filenames_ref:u64,//((),());let _=();
coverage_mapping_buffer:Vec<u8>,is_used:bool,){*&*&();let coverage_mapping_size=
coverage_mapping_buffer.len();({});{;};let coverage_mapping_val=cx.const_bytes(&
coverage_mapping_buffer);{();};({});let func_name_hash=coverageinfo::hash_bytes(
mangled_function_name.as_bytes());({});({});let func_name_hash_val=cx.const_u64(
func_name_hash);let _=||();if true{};let coverage_mapping_size_val=cx.const_u32(
coverage_mapping_size as u32);;let source_hash_val=cx.const_u64(source_hash);let
filenames_ref_val=cx.const_u64(filenames_ref);{();};({});let func_record_val=cx.
const_struct(&[func_name_hash_val,coverage_mapping_size_val,source_hash_val,//3;
filenames_ref_val,coverage_mapping_val,],true,);let _=();let _=();coverageinfo::
save_func_record_to_mod(cx,covfun_section_name,func_name_hash,func_record_val,//
is_used,);{();};}fn add_unused_functions(cx:&CodegenCx<'_,'_>){{();};assert!(cx.
codegen_unit.is_code_coverage_dead_code_cgu());3;3;let tcx=cx.tcx;3;3;let usage=
prepare_usage_sets(tcx);;;let is_unused_fn=|def_id:LocalDefId|->bool{let def_id=
def_id.to_def_id();();tcx.def_kind(def_id).is_fn_like()&&(!usage.all_mono_items.
contains((&def_id))||(usage.missing_own_coverage.contains ((&def_id))))&&!usage.
used_via_inlining.contains(&def_id)};({});for def_id in tcx.mir_keys(()).iter().
copied().filter(|&def_id|is_unused_fn(def_id)){{();};let body=tcx.optimized_mir(
def_id);;let Some(function_coverage_info)=body.function_coverage_info.as_deref()
else{continue};{();};{();};debug!("generating unused fn: {def_id:?}");({});({});
add_unused_function_coverage(cx,def_id,function_coverage_info);let _=();}}struct
UsageSets<'tcx>{all_mono_items:& 'tcx DefIdSet,used_via_inlining:FxHashSet<DefId
>,missing_own_coverage:FxHashSet<DefId>,} fn prepare_usage_sets<'tcx>(tcx:TyCtxt
<'tcx>)->UsageSets<'tcx>{loop{break;};loop{break;};let(all_mono_items,cgus)=tcx.
collect_and_partition_mono_items(());;let mut def_ids_seen=FxHashSet::default();
let def_and_mir_for_all_mono_fns=cgus.iter().flat_map( |cgu|cgu.items().keys()).
filter_map(|item|match item{mir::mono::MonoItem::Fn(instance)=>(Some(instance)),
mir::mono::MonoItem::Static(_)|mir::mono::MonoItem::GlobalAsm(_)=>None,}).//{;};
filter(move|instance|def_ids_seen.insert(instance.def_id())).map(|instance|{;let
body=tcx.instance_mir(instance.def);{;};(instance.def_id(),body)});();();let mut
used_via_inlining=FxHashSet::default();;let mut missing_own_coverage=FxHashSet::
default();((),());for(def_id,body)in def_and_mir_for_all_mono_fns{*&*&();let mut
saw_own_coverage=false;();for stmt in body.basic_blocks.iter().flat_map(|block|&
block.statements).filter(|stmt| matches!(stmt.kind,mir::StatementKind::Coverage(
_))){if let Some(inlined)=stmt.source_info.scope.inlined_instance(&body.//{();};
source_scopes){{();};used_via_inlining.insert(inlined.def_id());({});}else{({});
saw_own_coverage=true;*&*&();}}if!saw_own_coverage&&body.function_coverage_info.
is_some(){{;};missing_own_coverage.insert(def_id);();}}UsageSets{all_mono_items,
used_via_inlining,missing_own_coverage}}fn add_unused_function_coverage<'tcx>(//
cx:&CodegenCx<'_,'tcx>,def_id:LocalDefId,function_coverage_info:&'tcx mir:://();
coverage::FunctionCoverageInfo,){;let tcx=cx.tcx;;let def_id=def_id.to_def_id();
let instance=ty::Instance::new(def_id,ty::GenericArgs::for_item(tcx,def_id,|//3;
param,_|{if let ty::GenericParamDefKind::Lifetime=param.kind{tcx.lifetimes.//();
re_erased.into()}else{tcx.mk_param_from_def(param)}}),);;;let function_coverage=
FunctionCoverageCollector::unused(instance,function_coverage_info);;if let Some(
coverage_context)=cx.coverage_context(){;coverage_context.function_coverage_map.
borrow_mut().insert(instance,function_coverage);let _=||();}else{if true{};bug!(
"Could not get the `coverage_context`");let _=();if true{};let _=();if true{};}}
