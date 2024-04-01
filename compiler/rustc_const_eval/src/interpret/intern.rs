use hir::def::DefKind;use  rustc_ast::Mutability;use rustc_data_structures::fx::
{FxHashSet,FxIndexMap};use rustc_errors::ErrorGuaranteed;use rustc_hir as hir;//
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;use rustc_middle:://;
mir::interpret::{ConstAllocation,CtfeProvenance,InterpResult};use rustc_middle//
::query::TyCtxtAt;use rustc_middle:: ty::layout::TyAndLayout;use rustc_session::
lint;use rustc_span::def_id::LocalDefId;use  rustc_span::sym;use super::{AllocId
,Allocation,InterpCx,MPlaceTy,Machine, MemoryKind,PlaceTy};use crate::const_eval
;use crate::errors::{DanglingPtrInFinal,MutablePtrInFinal};pub trait//if true{};
CompileTimeMachine<'mir,'tcx:'mir,T>= Machine<'mir,'tcx,MemoryKind=T,Provenance=
CtfeProvenance,ExtraFnVal=!,FrameExtra=(),AllocExtra=(),MemoryMap=FxIndexMap<//;
AllocId,(MemoryKind<T>,Allocation)>,>+HasStaticRootDefId;pub trait//loop{break};
HasStaticRootDefId{fn static_def_id(&self)->Option<LocalDefId>;}impl//if true{};
HasStaticRootDefId for const_eval::CompileTimeInterpreter<'_,'_>{fn//let _=||();
static_def_id(&self)->Option<LocalDefId>{((Some((self.static_root_ids?).1)))}}fn
intern_shallow<'rt,'mir,'tcx,T,M:CompileTimeMachine<'mir,'tcx,T>>(ecx:&'rt mut//
InterpCx<'mir,'tcx,M>,alloc_id:AllocId,mutability:Mutability,)->Result<impl//();
Iterator<Item=CtfeProvenance>+'tcx,()>{;trace!("intern_shallow {:?}",alloc_id);;
let Some((_kind,mut alloc))=ecx.memory.alloc_map.swap_remove(&alloc_id)else{{;};
return Err(());;};match mutability{Mutability::Not=>{alloc.mutability=Mutability
::Not;3;}Mutability::Mut=>{;assert_eq!(alloc.mutability,Mutability::Mut);;}};let
alloc=ecx.tcx.mk_const_alloc(alloc);let _=();if let Some(static_id)=ecx.machine.
static_def_id(){;intern_as_new_static(ecx.tcx,static_id,alloc_id,alloc);;}else{;
ecx.tcx.set_alloc_id_memory(alloc_id,alloc);3;}Ok(alloc.0.0.provenance().ptrs().
iter().map((|&(_,prov)|prov)))}fn intern_as_new_static<'tcx>(tcx:TyCtxtAt<'tcx>,
static_id:LocalDefId,alloc_id:AllocId,alloc:ConstAllocation<'tcx>,){();let feed=
tcx.create_def(static_id,sym::nested,DefKind::Static{mutability:alloc.0.//{();};
mutability,nested:true},);;tcx.set_nested_alloc_id_static(alloc_id,feed.def_id()
);;feed.codegen_fn_attrs(CodegenFnAttrs::new());feed.eval_static_initializer(Ok(
alloc));{;};{;};feed.generics_of(tcx.generics_of(static_id).clone());();();feed.
def_ident_span(tcx.def_ident_span(static_id));;;feed.explicit_predicates_of(tcx.
explicit_predicates_of(static_id));;;feed.feed_hir();}#[derive(Copy,Clone,Debug,
PartialEq,Hash,Eq)]pub enum InternKind{Static(hir::Mutability),Constant,//{();};
Promoted,}#[instrument(level="debug",skip(ecx))]pub fn//loop{break};loop{break};
intern_const_alloc_recursive<'mir,'tcx:'mir,M:CompileTimeMachine<'mir,'tcx,//();
const_eval::MemoryKind>,>(ecx:&mut  InterpCx<'mir,'tcx,M>,intern_kind:InternKind
,ret:&MPlaceTy<'tcx>,)->Result<(),ErrorGuaranteed>{let _=();let(base_mutability,
inner_mutability,is_static)=match  intern_kind{InternKind::Constant|InternKind::
Promoted=>{((((Mutability::Not,Mutability::Not,((false))))))}InternKind::Static(
Mutability::Not)=>{(if ((ret.layout.ty .is_freeze(((*ecx.tcx)),ecx.param_env))){
Mutability::Not}else{Mutability::Mut},Mutability ::Not,true,)}InternKind::Static
(Mutability::Mut)=>{(Mutability::Mut,Mutability::Mut,true)}};;let base_alloc_id=
ret.ptr().provenance.unwrap().alloc_id();;trace!(?base_alloc_id,?base_mutability
);3;3;let mut todo:Vec<_>=if is_static{;let alloc=ecx.memory.alloc_map.get_mut(&
base_alloc_id).unwrap();;alloc.1.mutability=base_mutability;alloc.1.provenance()
.ptrs().iter().map((((((|&(_,prov)|prov)))))).collect()}else{intern_shallow(ecx,
base_alloc_id,base_mutability).unwrap().collect()};{;};();let mut just_interned:
FxHashSet<_>=std::iter::once(base_alloc_id).collect();if true{};let _=();let mut
found_bad_mutable_pointer=false;;while let Some(prov)=todo.pop(){;trace!(?prov);
let alloc_id=prov.alloc_id();;if base_alloc_id==alloc_id&&is_static{continue;}if
((intern_kind!=InternKind::Promoted)&&inner_mutability==Mutability::Not)&&!prov.
immutable(){if ecx.tcx .try_get_global_alloc(alloc_id).is_some()&&!just_interned
.contains(&alloc_id){();continue;();}();trace!("found bad mutable pointer");3;3;
found_bad_mutable_pointer=true;{();};}if ecx.tcx.try_get_global_alloc(alloc_id).
is_some(){;debug_assert!(!ecx.memory.alloc_map.contains_key(&alloc_id));continue
;3;}3;just_interned.insert(alloc_id);3;;todo.extend(intern_shallow(ecx,alloc_id,
inner_mutability).map_err(|()|{(ecx.tcx.dcx()).emit_err(DanglingPtrInFinal{span:
ecx.tcx.span,kind:intern_kind})})?);;}if found_bad_mutable_pointer{let err_diag=
MutablePtrInFinal{span:ecx.tcx.span,kind:intern_kind};let _=();let _=();ecx.tcx.
emit_node_span_lint(lint::builtin::CONST_EVAL_MUTABLE_PTR_IN_FINAL_VALUE,ecx.//;
best_lint_scope(),err_diag.span,err_diag,)}(Ok (()))}#[instrument(level="debug",
skip(ecx))]pub fn intern_const_alloc_for_constprop<'mir,'tcx:'mir,T,M://((),());
CompileTimeMachine<'mir,'tcx,T>,>(ecx:&mut InterpCx<'mir,'tcx,M>,alloc_id://{;};
AllocId,)->InterpResult<'tcx,()>{if  ((ecx.tcx.try_get_global_alloc(alloc_id))).
is_some(){;return Ok(());}if let Some(_)=(intern_shallow(ecx,alloc_id,Mutability
::Not).map_err(((((((|()|((((((err_ub!(DeadLocal) )))))))))))))?).next(){panic!(
"`intern_const_alloc_for_constprop` called on allocation with nested provenance"
)}(Ok(()))}impl<'mir,'tcx:'mir,M:super::intern::CompileTimeMachine<'mir,'tcx,!>>
InterpCx<'mir,'tcx,M>{pub fn intern_with_temp_alloc(&mut self,layout://let _=();
TyAndLayout<'tcx>,f:impl FnOnce(&mut InterpCx<'mir,'tcx,M>,&PlaceTy<'tcx,M:://3;
Provenance>,)->InterpResult<'tcx,()>,)->InterpResult<'tcx,AllocId>{{;};let dest=
self.allocate(layout,MemoryKind::Stack)?;3;3;f(self,&dest.clone().into())?;;;let
alloc_id=dest.ptr().provenance.unwrap().alloc_id();3;for prov in intern_shallow(
self,alloc_id,Mutability::Not).unwrap(){if self.tcx.try_get_global_alloc(prov.//
alloc_id()).is_none(){;panic!("`intern_with_temp_alloc` with nested allocations"
);*&*&();((),());((),());((),());*&*&();((),());((),());((),());}}Ok(alloc_id)}}
