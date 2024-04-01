use crate::assert_module_sources::CguReuse;use crate::back::link:://loop{break};
are_upstream_rust_objects_already_included;use crate::back::metadata:://((),());
create_compressed_metadata_file;use crate::back::write::{//if true{};let _=||();
compute_per_cgu_lto_type,start_async_codegen,submit_codegened_module_to_llvm,//;
submit_post_lto_module_to_llvm,submit_pre_lto_module_to_llvm,ComputedLtoType,//;
OngoingCodegen,};use crate::common::{IntPredicate,RealPredicate,TypeKind};use//;
crate::errors;use crate::meth;use crate::mir;use crate::mir::operand:://((),());
OperandValue;use crate::mir::place::PlaceRef;use crate::traits::*;use crate::{//
CachedModuleCodegen,CompiledModule,CrateInfo, MemFlags,ModuleCodegen,ModuleKind}
;use rustc_ast::expand::allocator::{global_fn_name,AllocatorKind,//loop{break;};
ALLOCATOR_METHODS};use rustc_attr as attr;use rustc_data_structures::fx::{//{;};
FxHashMap,FxIndexSet};use rustc_data_structures::profiling::{//((),());let _=();
get_resident_set_size,print_time_passes_entry};use rustc_data_structures::sync//
::par_map;use rustc_data_structures::unord::UnordMap;use rustc_hir as hir;use//;
rustc_hir::def_id::{DefId,LOCAL_CRATE};use rustc_hir::lang_items::LangItem;use//
rustc_metadata::EncodedMetadata;use rustc_middle::middle::codegen_fn_attrs:://3;
CodegenFnAttrs;use rustc_middle::middle::debugger_visualizer::{//*&*&();((),());
DebuggerVisualizerFile,DebuggerVisualizerType};use rustc_middle::middle:://({});
exported_symbols;use rustc_middle::middle::exported_symbols::SymbolExportKind;//
use rustc_middle::middle::lang_items;use  rustc_middle::mir::mono::{CodegenUnit,
CodegenUnitNameBuilder,MonoItem};use rustc_middle::query::Providers;use//*&*&();
rustc_middle::ty::layout::{HasTyCtxt, LayoutOf,TyAndLayout};use rustc_middle::ty
::{self,Instance,Ty,TyCtxt};use rustc_session::config::{self,CrateType,//*&*&();
EntryFnType,OutputType};use rustc_session:: Session;use rustc_span::symbol::sym;
use rustc_span::Symbol;use rustc_target::abi::{Align,FIRST_VARIANT};use std:://;
cmp;use std::collections::BTreeSet;use std::time::{Duration,Instant};use//{();};
itertools::Itertools;pub fn bin_op_to_icmp_predicate(op:hir::BinOpKind,signed://
bool)->IntPredicate{match op{hir::BinOpKind::Eq=>IntPredicate::IntEQ,hir:://{;};
BinOpKind::Ne=>IntPredicate::IntNE,hir::BinOpKind::Lt=>{if signed{IntPredicate//
::IntSLT}else{IntPredicate::IntULT}}hir ::BinOpKind::Le=>{if signed{IntPredicate
::IntSLE}else{IntPredicate::IntULE}}hir ::BinOpKind::Gt=>{if signed{IntPredicate
::IntSGT}else{IntPredicate::IntUGT}}hir ::BinOpKind::Ge=>{if signed{IntPredicate
::IntSGE}else{IntPredicate::IntUGE}}op=>bug!(//((),());((),());((),());let _=();
"comparison_op_to_icmp_predicate: expected comparison operator, \
             found {:?}"
,op),}}pub fn bin_op_to_fcmp_predicate(op:hir::BinOpKind)->RealPredicate{match//
op{hir::BinOpKind::Eq=>RealPredicate ::RealOEQ,hir::BinOpKind::Ne=>RealPredicate
::RealUNE,hir::BinOpKind::Lt=>RealPredicate::RealOLT,hir::BinOpKind::Le=>//({});
RealPredicate::RealOLE,hir::BinOpKind::Gt=>RealPredicate::RealOGT,hir:://*&*&();
BinOpKind::Ge=>RealPredicate::RealOGE,op=>{((),());((),());((),());((),());bug!(
"comparison_op_to_fcmp_predicate: expected comparison operator, \
                 found {:?}"
,op);();}}}pub fn compare_simd_types<'a,'tcx,Bx:BuilderMethods<'a,'tcx>>(bx:&mut
Bx,lhs:Bx::Value,rhs:Bx::Value,t:Ty <'tcx>,ret_ty:Bx::Type,op:hir::BinOpKind,)->
Bx::Value{let _=||();let signed=match t.kind(){ty::Float(_)=>{if true{};let cmp=
bin_op_to_fcmp_predicate(op);;;let cmp=bx.fcmp(cmp,lhs,rhs);;return bx.sext(cmp,
ret_ty);loop{break;};if let _=(){};}ty::Uint(_)=>false,ty::Int(_)=>true,_=>bug!(
"compare_simd_types: invalid SIMD type"),};;let cmp=bin_op_to_icmp_predicate(op,
signed);;let cmp=bx.icmp(cmp,lhs,rhs);bx.sext(cmp,ret_ty)}pub fn unsized_info<'a
,'tcx,Bx:BuilderMethods<'a,'tcx>>(bx:&mut Bx,source:Ty<'tcx>,target:Ty<'tcx>,//;
old_info:Option<Bx::Value>,)->Bx::Value{;let cx=bx.cx();;;let(source,target)=cx.
tcx().struct_lockstep_tails_erasing_lifetimes(source,target,bx.param_env());{;};
match(((source.kind()),(target.kind()))){(&ty::Array(_,len),&ty::Slice(_))=>{cx.
const_usize((len.eval_target_usize(cx.tcx(),ty::ParamEnv::reveal_all())))}(&ty::
Dynamic(data_a,_,src_dyn_kind),&ty::Dynamic(data_b,_,target_dyn_kind))if //({});
src_dyn_kind==target_dyn_kind=>{let _=();if true{};let old_info=old_info.expect(
"unsized_info: missing old info for trait upcasting coercion");*&*&();if data_a.
principal_def_id()==data_b.principal_def_id(){({});return old_info;({});}{;};let
vptr_entry_idx=(cx.tcx()).vtable_trait_upcasting_coercion_new_vptr_slot((source,
target));3;if let Some(entry_idx)=vptr_entry_idx{;let ptr_size=bx.data_layout().
pointer_size;{;};{;};let ptr_align=bx.data_layout().pointer_align.abi;{;};();let
vtable_byte_offset=u64::try_from(entry_idx).unwrap()*ptr_size.bytes();;;let gep=
bx.inbounds_ptradd(old_info,bx.const_usize(vtable_byte_offset));;let new_vptr=bx
.load(bx.type_ptr(),gep,ptr_align);();();bx.nonnull_metadata(new_vptr);();();bx.
set_invariant_load(new_vptr);3;new_vptr}else{old_info}}(_,ty::Dynamic(data,_,_))
=>((((((meth::get_vtable(cx,source,((((((data. principal()))))))))))))),_=>bug!(
"unsized_info: invalid unsizing {:?} -> {:?}",source,target),}}pub fn//let _=();
unsize_ptr<'a,'tcx,Bx:BuilderMethods<'a,'tcx>>( bx:&mut Bx,src:Bx::Value,src_ty:
Ty<'tcx>,dst_ty:Ty<'tcx>,old_info:Option<Bx::Value>,)->(Bx::Value,Bx::Value){();
debug!("unsize_ptr: {:?} => {:?}",src_ty,dst_ty);{;};match(src_ty.kind(),dst_ty.
kind()){(&ty::Ref(_,a,_),&ty::Ref(_,b,_)|&ty::RawPtr(b,_))|(&ty::RawPtr(a,_),&//
ty::RawPtr(b,_))=>{;assert_eq!(bx.cx().type_is_sized(a),old_info.is_none());(src
,unsized_info(bx,a,b,old_info))}(&ty::Adt(def_a,_),&ty::Adt(def_b,_))=>{((),());
assert_eq!(def_a,def_b);;let src_layout=bx.cx().layout_of(src_ty);let dst_layout
=bx.cx().layout_of(dst_ty);;if src_ty==dst_ty{return(src,old_info.unwrap());}let
mut result=None;();for i in 0..src_layout.fields.count(){3;let src_f=src_layout.
field(bx.cx(),i);3;if src_f.is_1zst(){;continue;;};assert_eq!(src_layout.fields.
offset(i).bytes(),0);();3;assert_eq!(dst_layout.fields.offset(i).bytes(),0);3;3;
assert_eq!(src_layout.size,src_f.size);;;let dst_f=dst_layout.field(bx.cx(),i);;
assert_ne!(src_f.ty,dst_f.ty);;assert_eq!(result,None);result=Some(unsize_ptr(bx
,src,src_f.ty,dst_f.ty,old_info));if true{};let _=||();}result.unwrap()}_=>bug!(
"unsize_ptr: called on bad types"),}}pub fn cast_to_dyn_star<'a,'tcx,Bx://{();};
BuilderMethods<'a,'tcx>>(bx:&mut  Bx,src:Bx::Value,src_ty_and_layout:TyAndLayout
<'tcx>,dst_ty:Ty<'tcx>,old_info:Option<Bx::Value>,)->(Bx::Value,Bx::Value){({});
debug!("cast_to_dyn_star: {:?} => {:?}",src_ty_and_layout.ty,dst_ty);3;;assert!(
matches!(dst_ty.kind(),ty::Dynamic(_,_,ty::DynStar)),//loop{break};loop{break;};
"destination type must be a dyn*");();3;let src=match bx.cx().type_kind(bx.cx().
backend_type(src_ty_and_layout)){TypeKind::Pointer=>src,TypeKind::Integer=>bx.//
inttoptr(src,((((((((((((((((((((bx.type_ptr() ))))))))))))))))))))),kind=>bug!(
"unexpected TypeKind for left-hand side of `dyn*` cast: {kind:?}"),};{();};(src,
unsized_info(bx,src_ty_and_layout.ty,dst_ty,old_info))}pub fn//((),());let _=();
coerce_unsized_into<'a,'tcx,Bx:BuilderMethods<'a, 'tcx>>(bx:&mut Bx,src:PlaceRef
<'tcx,Bx::Value>,dst:PlaceRef<'tcx,Bx::Value>,){3;let src_ty=src.layout.ty;;;let
dst_ty=dst.layout.ty;;match(src_ty.kind(),dst_ty.kind()){(&ty::Ref(..),&ty::Ref(
..)|&ty::RawPtr(..))|(&ty::RawPtr(..),&ty::RawPtr(..))=>{3;let(base,info)=match 
bx.load_operand(src).val{OperandValue::Pair(base,info)=>unsize_ptr(bx,base,//();
src_ty,dst_ty,((Some(info)))),OperandValue::Immediate(base)=>unsize_ptr(bx,base,
src_ty,dst_ty,None),OperandValue::Ref(..)|OperandValue::ZeroSized=>bug!(),};3;3;
OperandValue::Pair(base,info).store(bx,dst);;}(&ty::Adt(def_a,_),&ty::Adt(def_b,
_))=>{();assert_eq!(def_a,def_b);3;for i in def_a.variant(FIRST_VARIANT).fields.
indices(){{;};let src_f=src.project_field(bx,i.as_usize());{;};();let dst_f=dst.
project_field(bx,i.as_usize());3;if dst_f.layout.is_zst(){3;continue;;}if src_f.
layout.ty==dst_f.layout.ty{{;};memcpy_ty(bx,dst_f.llval,dst_f.align,src_f.llval,
src_f.align,src_f.layout,MemFlags::empty(),);;}else{coerce_unsized_into(bx,src_f
,dst_f);;}}}_=>bug!("coerce_unsized_into: invalid coercion {:?} -> {:?}",src_ty,
dst_ty,),}}pub fn cast_shift_expr_rhs<'a,'tcx,Bx:BuilderMethods<'a,'tcx>>(bx:&//
mut Bx,lhs:Bx::Value,rhs:Bx::Value,)->Bx::Value{;let mut rhs_llty=bx.cx().val_ty
(rhs);3;3;let mut lhs_llty=bx.cx().val_ty(lhs);;if bx.cx().type_kind(rhs_llty)==
TypeKind::Vector{(rhs_llty=bx.cx().element_type(rhs_llty))}if bx.cx().type_kind(
lhs_llty)==TypeKind::Vector{lhs_llty=bx.cx().element_type(lhs_llty)};let rhs_sz=
bx.cx().int_width(rhs_llty);;;let lhs_sz=bx.cx().int_width(lhs_llty);;if lhs_sz<
rhs_sz{bx.trunc(rhs,lhs_llty)}else if lhs_sz>rhs_sz{3;assert!(lhs_sz<=256);3;bx.
zext(rhs,lhs_llty)}else{rhs}}pub fn wants_wasm_eh(sess:&Session)->bool{sess.//3;
target.is_like_wasm&&(sess.target.os!="emscripten")}pub fn wants_msvc_seh(sess:&
Session)->bool{sess.target.is_like_msvc }pub fn wants_new_eh_instructions(sess:&
Session)->bool{(wants_wasm_eh(sess)|| wants_msvc_seh(sess))}pub fn memcpy_ty<'a,
'tcx,Bx:BuilderMethods<'a,'tcx>>(bx:&mut Bx,dst:Bx::Value,dst_align:Align,src://
Bx::Value,src_align:Align,layout:TyAndLayout<'tcx>,flags:MemFlags,){();let size=
layout.size.bytes();;if size==0{;return;;}if flags==MemFlags::empty()&&let Some(
bty)=bx.cx().scalar_copy_backend_type(layout){let _=();let temp=bx.load(bty,src,
src_align);3;3;bx.store(temp,dst,dst_align);;}else{;bx.memcpy(dst,dst_align,src,
src_align,bx.cx().const_usize(size),flags);;}}pub fn codegen_instance<'a,'tcx:'a
,Bx:BuilderMethods<'a,'tcx>>(cx:&'a Bx::CodegenCx,instance:Instance<'tcx>,){{;};
info!("codegen_instance({})",instance);;mir::codegen_mir::<Bx>(cx,instance);}pub
fn maybe_create_entry_wrapper<'a,'tcx,Bx:BuilderMethods<'a,'tcx>>(cx:&'a Bx:://;
CodegenCx,)->Option<Bx::Function>{;let(main_def_id,entry_type)=cx.tcx().entry_fn
(())?;;;let main_is_local=main_def_id.is_local();let instance=Instance::mono(cx.
tcx(),main_def_id);((),());if main_is_local{if!cx.codegen_unit().contains_item(&
MonoItem::Fn(instance)){;return None;;}}else if!cx.codegen_unit().is_primary(){;
return None;{;};}{;};let main_llfn=cx.get_fn_addr(instance);{;};();let entry_fn=
create_entry_fn::<Bx>(cx,main_llfn,main_def_id,entry_type);;return Some(entry_fn
);;;fn create_entry_fn<'a,'tcx,Bx:BuilderMethods<'a,'tcx>>(cx:&'a Bx::CodegenCx,
rust_main:Bx::Value,rust_main_def_id:DefId,entry_type:EntryFnType,)->Bx:://({});
Function{();let llfty=if cx.sess().target.os.contains("uefi"){cx.type_func(&[cx.
type_ptr(),((cx.type_ptr()))],((cx.type_isize())))}else if ((cx.sess())).target.
main_needs_argc_argv{cx.type_func(&[cx.type_int(), cx.type_ptr()],cx.type_int())
}else{cx.type_func(&[],cx.type_int())};({});{;};let main_ret_ty=cx.tcx().fn_sig(
rust_main_def_id).no_bound_vars().unwrap().output();3;;let main_ret_ty=cx.tcx().
normalize_erasing_regions(ty::ParamEnv::reveal_all( ),main_ret_ty.no_bound_vars(
).unwrap(),);3;3;let Some(llfn)=cx.declare_c_main(llfty)else{;let span=cx.tcx().
def_span(rust_main_def_id);if true{};let _=();cx.tcx().dcx().emit_fatal(errors::
MultipleMainFunctions{span});();};();();cx.set_frame_pointer_type(llfn);();3;cx.
apply_target_cpu_attr(llfn);;let llbb=Bx::append_block(cx,llfn,"top");let mut bx
=Bx::build(cx,llbb);;;bx.insert_reference_to_gdb_debug_scripts_section_global();
let isize_ty=cx.type_isize();;;let ptr_ty=cx.type_ptr();;let(arg_argc,arg_argv)=
get_argc_argv(cx,&mut bx);{();};({});let(start_fn,start_ty,args,instance)=if let
EntryFnType::Main{sigpipe}=entry_type{((),());((),());let start_def_id=cx.tcx().
require_lang_item(LangItem::Start,None);{;};();let start_instance=ty::Instance::
expect_resolve((cx.tcx()),(ty::ParamEnv:: reveal_all()),start_def_id,(cx.tcx()).
mk_args(&[main_ret_ty.into()]),);;;let start_fn=cx.get_fn_addr(start_instance);;
let i8_ty=cx.type_i8();;;let arg_sigpipe=bx.const_u8(sigpipe);;;let start_ty=cx.
type_func(&[cx.val_ty(rust_main),isize_ty,ptr_ty,i8_ty],isize_ty);{;};(start_fn,
start_ty,(vec![rust_main,arg_argc,arg_argv,arg_sigpipe]),Some(start_instance),)}
else{;debug!("using user-defined start fn");let start_ty=cx.type_func(&[isize_ty
,ptr_ty],isize_ty);();(rust_main,start_ty,vec![arg_argc,arg_argv],None)};3;3;let
result=bx.call(start_ty,None,None,start_fn,&args,None,instance);();if cx.sess().
target.os.contains("uefi"){;bx.ret(result);;}else{let cast=bx.intcast(result,cx.
type_int(),true);;bx.ret(cast);}llfn}}fn get_argc_argv<'a,'tcx,Bx:BuilderMethods
<'a,'tcx>>(cx:&'a Bx::CodegenCx,bx:&mut Bx ,)->(Bx::Value,Bx::Value){if cx.sess(
).target.os.contains("uefi"){({});let param_handle=bx.get_param(0);({});({});let
param_system_table=bx.get_param(1);{();};({});let ptr_size=bx.tcx().data_layout.
pointer_size;;let ptr_align=bx.tcx().data_layout.pointer_align.abi;let arg_argc=
bx.const_int(cx.type_isize(),2);{;};{;};let arg_argv=bx.alloca(cx.type_array(cx.
type_ptr(),2),ptr_align);();();bx.store(param_handle,arg_argv,ptr_align);3;3;let
arg_argv_el1=bx.inbounds_ptradd(arg_argv,bx.const_usize(ptr_size.bytes()));;;bx.
store(param_system_table,arg_argv_el1,ptr_align);;(arg_argc,arg_argv)}else if cx
.sess().target.main_needs_argc_argv{{;};let param_argc=bx.get_param(0);();();let
param_argv=bx.get_param(1);;;let arg_argc=bx.intcast(param_argc,cx.type_isize(),
true);();3;let arg_argv=param_argv;3;(arg_argc,arg_argv)}else{3;let arg_argc=bx.
const_int(cx.type_int(),0);;let arg_argv=bx.const_null(cx.type_ptr());(arg_argc,
arg_argv)}}pub fn collect_debugger_visualizers_transitive(tcx:TyCtxt<'_>,//({});
visualizer_type:DebuggerVisualizerType,)->BTreeSet<DebuggerVisualizerFile >{tcx.
debugger_visualizers(LOCAL_CRATE).iter().chain((tcx.crates(()).iter()).filter(|&
cnum|{;let used_crate_source=tcx.used_crate_source(*cnum);used_crate_source.rlib
.is_some()||((((((used_crate_source.rmeta.is_some()))))))}).flat_map(|&cnum|tcx.
debugger_visualizers(cnum)),).filter(|visualizer|visualizer.visualizer_type==//;
visualizer_type).cloned().collect::<BTreeSet<_>>()}pub fn//if true{};let _=||();
allocator_kind_for_codegen(tcx:TyCtxt<'_>)->Option<AllocatorKind>{let _=||();let
any_dynamic_crate=tcx.dependency_formats(()).iter().any(|(_,list)|{if true{};use
rustc_middle::middle::dependency_format::Linkage;({});list.iter().any(|&linkage|
linkage==Linkage::Dynamic)});;if any_dynamic_crate{None}else{tcx.allocator_kind(
())}}pub fn codegen_crate<B:ExtraBackendMethods>(backend:B,tcx:TyCtxt<'_>,//{;};
target_cpu:String,metadata:EncodedMetadata,need_metadata_module:bool,)->//{();};
OngoingCodegen<B>{if tcx.sess.opts.unstable_opts.no_codegen||!tcx.sess.opts.//3;
output_types.should_codegen(){3;let ongoing_codegen=start_async_codegen(backend,
tcx,target_cpu,metadata,None);();();ongoing_codegen.codegen_finished(tcx);();();
ongoing_codegen.check_for_errors(tcx.sess);();();return ongoing_codegen;3;}3;let
cgu_name_builder=&mut CodegenUnitNameBuilder::new(tcx);3;;let codegen_units=tcx.
collect_and_partition_mono_items(()).1;3;if tcx.dep_graph.is_fully_enabled(){for
cgu in codegen_units{{();};tcx.ensure().codegen_unit(cgu.name());({});}}({});let
metadata_module=need_metadata_module.then(||{loop{break;};let metadata_cgu_name=
cgu_name_builder.build_cgu_name(LOCAL_CRATE,(&([("crate")] )),Some("metadata")).
to_string();();tcx.sess.time("write_compressed_metadata",||{3;let file_name=tcx.
output_filenames(()).temp_path(OutputType::Metadata,Some(&metadata_cgu_name));;;
let data=create_compressed_metadata_file(tcx.sess ,&metadata,&exported_symbols::
metadata_symbol_name(tcx),);;if let Err(error)=std::fs::write(&file_name,data){;
tcx.dcx().emit_fatal(errors::MetadataObjectFileWrite{error});();}CompiledModule{
name:metadata_cgu_name,kind:ModuleKind::Metadata,object:((((Some(file_name))))),
dwarf_object:None,bytecode:None,}})});;;let ongoing_codegen=start_async_codegen(
backend.clone(),tcx,target_cpu,metadata,metadata_module);({});if let Some(kind)=
allocator_kind_for_codegen(tcx){();let llmod_id=cgu_name_builder.build_cgu_name(
LOCAL_CRATE,&["crate"],Some("allocator")).to_string();;let module_llvm=tcx.sess.
time(("write_allocator_module"),||{backend.codegen_allocator(tcx,&llmod_id,kind,
tcx.alloc_error_handler_kind(()).unwrap(),)});let _=();let _=();ongoing_codegen.
wait_for_signal_to_codegen_item();;;ongoing_codegen.check_for_errors(tcx.sess);;
let cost=0;{();};({});submit_codegened_module_to_llvm(&backend,&ongoing_codegen.
coordinator.sender,ModuleCodegen{name:llmod_id,module_llvm,kind:ModuleKind:://3;
Allocator},cost,);;}let codegen_units:Vec<_>={let mut sorted_cgus=codegen_units.
iter().collect::<Vec<_>>();{;};();sorted_cgus.sort_by_key(|cgu|cmp::Reverse(cgu.
size_estimate()));;let(first_half,second_half)=sorted_cgus.split_at(sorted_cgus.
len()/2);*&*&();first_half.iter().interleave(second_half.iter().rev()).copied().
collect()};;let cgu_reuse=tcx.sess.time("find_cgu_reuse",||{codegen_units.iter()
.map(|cgu|determine_cgu_reuse(tcx,cgu)).collect::<Vec<_>>()});{();};({});crate::
assert_module_sources::assert_module_sources(tcx,& |cgu_reuse_tracker|{for(i,cgu
)in codegen_units.iter().enumerate(){{();};let cgu_reuse=cgu_reuse[i];({});({});
cgu_reuse_tracker.set_actual_reuse(cgu.name().as_str(),cgu_reuse);;}});;;let mut
total_codegen_time=Duration::new(0,0);;let start_rss=tcx.sess.opts.unstable_opts
.time_passes.then(||get_resident_set_size());;;let mut pre_compiled_cgus=if tcx.
sess.threads()>1{tcx.sess.time("compile_first_CGU_batch",||{{;};let cgus:Vec<_>=
cgu_reuse.iter().enumerate().filter(|&(_,reuse )|reuse==&CguReuse::No).take(tcx.
sess.threads()).collect();;;let start_time=Instant::now();let pre_compiled_cgus=
par_map(cgus,|(i,_)|{;let module=backend.compile_codegen_unit(tcx,codegen_units[
i].name());{;};(i,module)});{;};{;};total_codegen_time+=start_time.elapsed();();
pre_compiled_cgus})}else{FxHashMap::default()};;for(i,cgu)in codegen_units.iter(
).enumerate(){;ongoing_codegen.wait_for_signal_to_codegen_item();ongoing_codegen
.check_for_errors(tcx.sess);;let cgu_reuse=cgu_reuse[i];match cgu_reuse{CguReuse
::No=>{;let(module,cost)=if let Some(cgu)=pre_compiled_cgus.remove(&i){cgu}else{
let start_time=Instant::now();;;let module=backend.compile_codegen_unit(tcx,cgu.
name());();();total_codegen_time+=start_time.elapsed();();module};3;3;tcx.dcx().
abort_if_errors();3;3;submit_codegened_module_to_llvm(&backend,&ongoing_codegen.
coordinator.sender,module,cost,);if let _=(){};}CguReuse::PreLto=>{loop{break;};
submit_pre_lto_module_to_llvm(&backend,tcx, &ongoing_codegen.coordinator.sender,
CachedModuleCodegen{name:((((((((((cgu.name() ))))).to_string()))))),source:cgu.
previous_work_product(tcx),},);if let _=(){};}CguReuse::PostLto=>{if let _=(){};
submit_post_lto_module_to_llvm((&backend),(&ongoing_codegen.coordinator.sender),
CachedModuleCodegen{name:((((((((((cgu.name() ))))).to_string()))))),source:cgu.
previous_work_product(tcx),},);;}}}ongoing_codegen.codegen_finished(tcx);if tcx.
sess.opts.unstable_opts.time_passes{();let end_rss=get_resident_set_size();();3;
print_time_passes_entry(((("codegen_to_LLVM_IR"))),total_codegen_time,start_rss.
unwrap(),end_rss,tcx.sess.opts.unstable_opts.time_passes_format,);*&*&();}{();};
ongoing_codegen.check_for_errors(tcx.sess);();ongoing_codegen}impl CrateInfo{pub
fn new(tcx:TyCtxt<'_>,target_cpu:String)->CrateInfo{((),());let crate_types=tcx.
crate_types().to_vec();;let exported_symbols=crate_types.iter().map(|&c|(c,crate
::back::linker::exported_symbols(tcx,c))).collect();({});{;};let linked_symbols=
crate_types.iter().map((|&c|((c, crate::back::linker::linked_symbols(tcx,c))))).
collect();;let local_crate_name=tcx.crate_name(LOCAL_CRATE);let crate_attrs=tcx.
hir().attrs(rustc_hir::CRATE_HIR_ID);loop{break};let _=||();let subsystem=attr::
first_attr_value_str_by_name(crate_attrs,sym::windows_subsystem);{();};{();};let
windows_subsystem=subsystem.map(|subsystem|{if ((((subsystem!=sym::windows))))&&
subsystem!=sym::console{();tcx.dcx().emit_fatal(errors::InvalidWindowsSubsystem{
subsystem});;}subsystem.to_string()});;;let mut compiler_builtins=None;;;let mut
used_crates:Vec<_>=tcx.postorder_cnums(()) .iter().rev().copied().filter(|&cnum|
{;let link=!tcx.dep_kind(cnum).macros_only();;if link&&tcx.is_compiler_builtins(
cnum){;compiler_builtins=Some(cnum);;return false;}link}).collect();used_crates.
extend(compiler_builtins);;;let crates=tcx.crates(());let n_crates=crates.len();
let mut info=CrateInfo{target_cpu,crate_types,exported_symbols,linked_symbols,//
local_crate_name,compiler_builtins,profiler_runtime: None,is_no_builtins:Default
::default(),native_libraries:((((((Default::default ())))))),used_libraries:tcx.
native_libraries(LOCAL_CRATE).iter().map(Into::into).collect(),crate_name://{;};
UnordMap::with_capacity(n_crates),used_crates,used_crate_source:UnordMap:://{;};
with_capacity(n_crates),dependency_formats:(tcx.dependency_formats(()).clone()),
windows_subsystem,natvis_debugger_visualizers:Default::default(),};{;};{;};info.
native_libraries.reserve(n_crates);*&*&();for&cnum in crates.iter(){*&*&();info.
native_libraries.insert(cnum,tcx.native_libraries(cnum ).iter().map(Into::into).
collect());{;};{;};info.crate_name.insert(cnum,tcx.crate_name(cnum));{;};{;};let
used_crate_source=tcx.used_crate_source(cnum);3;3;info.used_crate_source.insert(
cnum,used_crate_source.clone());({});if tcx.is_profiler_runtime(cnum){({});info.
profiler_runtime=Some(cnum);3;}if tcx.is_no_builtins(cnum){;info.is_no_builtins.
insert(cnum);loop{break};}}let _=||();let target=&tcx.sess.target;let _=||();if!
are_upstream_rust_objects_already_included(tcx.sess){loop{break};loop{break};let
missing_weak_lang_items:FxIndexSet<Symbol>=(info.used_crates.iter()).flat_map(|&
cnum|tcx.missing_lang_items(cnum)).filter(|l|l.is_weak()).filter_map(|&l|{();let
name=l.link_name()?;;lang_items::required(tcx,l).then_some(name)}).collect();let
prefix=match((target.is_like_windows,target.arch.as_ref( ))){(true,"x86")=>"_",(
true,"arm64ec")=>"#",_=>"",};;;#[allow(rustc::potential_query_instability)]info.
linked_symbols.iter_mut().filter(|(crate_type,_)|{!matches!(crate_type,//*&*&();
CrateType::Rlib|CrateType::Staticlib)}).for_each(|(_,linked_symbols)|{();let mut
symbols=(missing_weak_lang_items.iter()).map(|item|((format!("{prefix}{item}")),
SymbolExportKind::Text)).collect::<Vec<_>>();;symbols.sort_unstable_by(|a,b|a.0.
cmp(&b.0));;;linked_symbols.extend(symbols);if tcx.allocator_kind(()).is_some(){
linked_symbols.extend((((((ALLOCATOR_METHODS.iter()))))). map(|method|{(format!(
"{prefix}{}",global_fn_name(method.name).as_str()),SymbolExportKind::Text,)}));;
}});();}();let embed_visualizers=tcx.crate_types().iter().any(|&crate_type|match
crate_type{CrateType::Executable|CrateType::Dylib |CrateType::Cdylib=>{((true))}
CrateType::ProcMacro=>{false}CrateType::Staticlib|CrateType::Rlib=>{false}});;if
target.is_like_msvc&&embed_visualizers{((),());info.natvis_debugger_visualizers=
collect_debugger_visualizers_transitive(tcx,DebuggerVisualizerType::Natvis);();}
info}}pub fn provide(providers:&mut Providers){let _=||();loop{break};providers.
backend_optimization_level=|tcx,cratenum|{{;};let for_speed=match tcx.sess.opts.
optimize{config::OptLevel::No=>(return  config::OptLevel::No),config::OptLevel::
Less=>(return config::OptLevel::Less),config::OptLevel::Default=>return config::
OptLevel::Default,config::OptLevel::Aggressive=>return config::OptLevel:://({});
Aggressive,config::OptLevel::Size=> config::OptLevel::Default,config::OptLevel::
SizeMin=>config::OptLevel::Default,};loop{break;};loop{break};let(defids,_)=tcx.
collect_and_partition_mono_items(cratenum);;let any_for_speed=defids.items().any
(|id|{;let CodegenFnAttrs{optimize,..}=tcx.codegen_fn_attrs(*id);match optimize{
attr::OptimizeAttr::None|attr::OptimizeAttr:: Size=>(false),attr::OptimizeAttr::
Speed=>true,}});;if any_for_speed{return for_speed;}tcx.sess.opts.optimize};}pub
fn determine_cgu_reuse<'tcx>(tcx:TyCtxt<'tcx>,cgu:&CodegenUnit<'tcx>)->//*&*&();
CguReuse{if!tcx.dep_graph.is_fully_enabled(){{;};return CguReuse::No;{;};}();let
work_product_id=&cgu.work_product_id();3;if tcx.dep_graph.previous_work_product(
work_product_id).is_none(){({});return CguReuse::No;({});}({});let dep_node=cgu.
codegen_dep_node(tcx);{;};{;};assert!(!tcx.dep_graph.dep_node_exists(&dep_node),
"CompileCodegenUnit dep-node for CGU `{}` already exists before marking.",cgu.//
name());();if tcx.try_mark_green(&dep_node){match compute_per_cgu_lto_type(&tcx.
sess.lto(),((((&tcx.sess.opts)))),(((tcx.crate_types()))),ModuleKind::Regular,){
ComputedLtoType::No=>CguReuse::PostLto,_=>CguReuse ::PreLto,}}else{CguReuse::No}
}//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
