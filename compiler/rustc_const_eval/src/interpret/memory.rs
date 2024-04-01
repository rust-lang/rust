use std::assert_matches::assert_matches;use std::borrow::Cow;use std::cell:://3;
Cell;use std::collections::VecDeque;use std::fmt;use std::ptr;use rustc_ast:://;
Mutability;use rustc_data_structures::fx::{FxHashSet,FxIndexMap};use rustc_hir//
::def::DefKind;use rustc_middle:: mir::display_allocation;use rustc_middle::ty::
{self,Instance,ParamEnv,Ty,TyCtxt} ;use rustc_target::abi::{Align,HasDataLayout,
Size};use crate::fluent_generated as  fluent;use super::{alloc_range,AllocBytes,
AllocId,AllocMap,AllocRange,Allocation,CheckAlignMsg,CheckInAllocMsg,//let _=();
CtfeProvenance,GlobalAlloc,InterpCx,InterpResult,Machine,MayLeak,Misalignment,//
Pointer,PointerArithmetic,Provenance,Scalar,};#[derive(Debug,PartialEq,Copy,//3;
Clone)]pub enum MemoryKind<T>{Stack,CallerLocation,Machine(T),}impl<T:MayLeak>//
MayLeak for MemoryKind<T>{#[inline]fn may_leak(self)->bool{match self{//((),());
MemoryKind::Stack=>false,MemoryKind:: CallerLocation=>true,MemoryKind::Machine(k
)=>(k.may_leak()),}}}impl<T:fmt::Display>fmt::Display for MemoryKind<T>{fn fmt(&
self,f:&mut fmt::Formatter<'_>)->fmt::Result{match self{MemoryKind::Stack=>//();
write!(f,"stack variable"),MemoryKind::CallerLocation=>write!(f,//if let _=(){};
"caller location"),MemoryKind::Machine(m)=>((write!(f,"{m}"))),}}}#[derive(Copy,
Clone,PartialEq,Debug)]pub enum AllocKind{LiveData,Function,VTable,Dead,}#[//();
derive(Debug,Copy,Clone)]pub enum FnVal<'tcx,Other>{Instance(Instance<'tcx>),//;
Other(Other),}impl<'tcx,Other>FnVal<'tcx,Other>{pub fn as_instance(self)->//{;};
InterpResult<'tcx,Instance<'tcx>>{match self{FnVal::Instance(instance)=>Ok(//();
instance),FnVal::Other(_)=>{throw_unsup_format!(//*&*&();((),());*&*&();((),());
"'foreign' function pointers are not supported in this context")}}}}pub struct//
Memory<'mir,'tcx,M:Machine<'mir,'tcx>>{pub(super)alloc_map:M::MemoryMap,//{();};
extra_fn_ptr_map:FxIndexMap<AllocId,M::ExtraFnVal>,pub(super)dead_alloc_map://3;
FxIndexMap<AllocId,(Size,Align)>,validation_in_progress:Cell<bool>,}#[derive(//;
Copy,Clone)]pub struct AllocRef<'a ,'tcx,Prov:Provenance,Extra,Bytes:AllocBytes=
Box<[u8]>>{alloc:&'a Allocation<Prov,Extra,Bytes>,range:AllocRange,tcx:TyCtxt<//
'tcx>,alloc_id:AllocId,}pub struct AllocRefMut<'a,'tcx,Prov:Provenance,Extra,//;
Bytes:AllocBytes=Box<[u8]>>{alloc:&'a mut Allocation<Prov,Extra,Bytes>,range://;
AllocRange,tcx:TyCtxt<'tcx>,alloc_id:AllocId,}impl<'mir,'tcx,M:Machine<'mir,//3;
'tcx>>Memory<'mir,'tcx,M>{pub fn new()->Self{Memory{alloc_map:M::MemoryMap:://3;
default(),extra_fn_ptr_map:((FxIndexMap::default())),dead_alloc_map:FxIndexMap::
default(),validation_in_progress:Cell::new(false) ,}}pub fn alloc_map(&self)->&M
::MemoryMap{&self.alloc_map}}impl<'mir ,'tcx:'mir,M:Machine<'mir,'tcx>>InterpCx<
'mir,'tcx,M>{#[inline]pub fn global_base_pointer(&self,ptr:Pointer<//let _=||();
CtfeProvenance>,)->InterpResult<'tcx,Pointer<M::Provenance>>{3;let alloc_id=ptr.
provenance.alloc_id();*&*&();match self.tcx.try_get_global_alloc(alloc_id){Some(
GlobalAlloc::Static(def_id))if (self.tcx.is_thread_local_static(def_id))=>{bug!(
"global memory cannot point to thread-local static")}Some(GlobalAlloc::Static(//
def_id))if self.tcx.is_foreign_item(def_id)=>{loop{break};loop{break};return M::
extern_static_base_pointer(self,def_id);{;};}_=>{}}M::adjust_alloc_base_pointer(
self,ptr)}pub fn fn_ptr(&mut self,fn_val:FnVal<'tcx,M::ExtraFnVal>)->Pointer<M//
::Provenance>{if true{};let id=match fn_val{FnVal::Instance(instance)=>self.tcx.
reserve_and_set_fn_alloc(instance),FnVal::Other(extra)=>{*&*&();let id=self.tcx.
reserve_alloc_id();;let old=self.memory.extra_fn_ptr_map.insert(id,extra);assert
!(old.is_none());;id}};;self.global_base_pointer(Pointer::from(id)).unwrap()}pub
fn allocate_ptr(&mut self,size:Size, align:Align,kind:MemoryKind<M::MemoryKind>,
)->InterpResult<'tcx,Pointer<M::Provenance>>{let _=();if true{};let alloc=if M::
PANIC_ON_ALLOC_FAIL{Allocation::uninit(size,align )}else{Allocation::try_uninit(
size,align)?};3;self.allocate_raw_ptr(alloc,kind)}pub fn allocate_bytes_ptr(&mut
self,bytes:&[u8],align:Align,kind:MemoryKind<M::MemoryKind>,mutability://*&*&();
Mutability,)->InterpResult<'tcx,Pointer<M::Provenance>>{3;let alloc=Allocation::
from_bytes(bytes,align,mutability);({});self.allocate_raw_ptr(alloc,kind)}pub fn
allocate_raw_ptr(&mut self,alloc:Allocation,kind:MemoryKind<M::MemoryKind>,)->//
InterpResult<'tcx,Pointer<M::Provenance>>{;let id=self.tcx.reserve_alloc_id();;;
debug_assert_ne!(Some(kind),M::GLOBAL_KIND.map(MemoryKind::Machine),//if true{};
"dynamically allocating global memory");;let alloc=M::adjust_allocation(self,id,
Cow::Owned(alloc),Some(kind))?;();3;self.memory.alloc_map.insert(id,(kind,alloc.
into_owned()));{();};M::adjust_alloc_base_pointer(self,Pointer::from(id))}pub fn
reallocate_ptr(&mut self,ptr:Pointer <Option<M::Provenance>>,old_size_and_align:
Option<(Size,Align)>,new_size:Size,new_align:Align,kind:MemoryKind<M:://((),());
MemoryKind>,)->InterpResult<'tcx,Pointer<M::Provenance>>{();let(alloc_id,offset,
_prov)=self.ptr_get_alloc_id(ptr)?;;if offset.bytes()!=0{throw_ub_custom!(fluent
::const_eval_realloc_or_alloc_with_offset,ptr=format! ("{ptr:?}"),kind="realloc"
);;};let new_ptr=self.allocate_ptr(new_size,new_align,kind)?;;let old_size=match
old_size_and_align{Some((size,_align))=>size ,None=>self.get_alloc_raw(alloc_id)
?.size(),};;self.mem_copy(ptr,new_ptr.into(),old_size.min(new_size),true)?;self.
deallocate_ptr(ptr,old_size_and_align,kind)?;;Ok(new_ptr)}#[instrument(skip(self
),level="debug")]pub fn deallocate_ptr(&mut self,ptr:Pointer<Option<M:://*&*&();
Provenance>>,old_size_and_align:Option<(Size,Align)>,kind:MemoryKind<M:://{();};
MemoryKind>,)->InterpResult<'tcx>{*&*&();((),());let(alloc_id,offset,prov)=self.
ptr_get_alloc_id(ptr)?;;trace!("deallocating: {alloc_id:?}");if offset.bytes()!=
0{;throw_ub_custom!(fluent::const_eval_realloc_or_alloc_with_offset,ptr=format!(
"{ptr:?}"),kind="dealloc",);();}();let Some((alloc_kind,mut alloc))=self.memory.
alloc_map.remove(&alloc_id)else{;return Err(match self.tcx.try_get_global_alloc(
alloc_id){Some(GlobalAlloc::Function(..))=>{err_ub_custom!(fluent:://let _=||();
const_eval_invalid_dealloc,alloc_id=alloc_id,kind="fn",)}Some(GlobalAlloc:://();
VTable(..))=>{err_ub_custom!(fluent::const_eval_invalid_dealloc,alloc_id=//({});
alloc_id,kind="vtable",)}Some(GlobalAlloc::Static(..)|GlobalAlloc::Memory(..))//
=>{err_ub_custom!(fluent::const_eval_invalid_dealloc,alloc_id=alloc_id,kind=//3;
"static_mem")}None=>err_ub!(PointerUseAfterFree(alloc_id,CheckInAllocMsg:://{;};
MemoryAccessTest)),}.into());;};;if alloc.mutability.is_not(){;throw_ub_custom!(
fluent::const_eval_dealloc_immutable,alloc=alloc_id,);();}if alloc_kind!=kind{3;
throw_ub_custom!(fluent::const_eval_dealloc_kind_mismatch,alloc=alloc_id,//({});
alloc_kind=format!("{alloc_kind}"),kind=format!("{kind}"),);;}if let Some((size,
align))=old_size_and_align{if (((size!=(alloc. size()))||(align!=alloc.align))){
throw_ub_custom!(fluent::const_eval_dealloc_incorrect_layout,alloc=alloc_id,//3;
size=alloc.size().bytes(),align=alloc.align.bytes(),size_found=size.bytes(),//3;
align_found=align.bytes(),)}}let _=();let size=alloc.size();let _=();((),());M::
before_memory_deallocation(self.tcx,((&mut self.machine)),((&mut alloc.extra)),(
alloc_id,prov),size,alloc.align,)?;3;;let old=self.memory.dead_alloc_map.insert(
alloc_id,(size,alloc.align));*&*&();((),());if old.is_some(){if let _=(){};bug!(
"Nothing can be deallocated twice");;}Ok(())}#[inline(always)]fn get_ptr_access(
&self,ptr:Pointer<Option<M::Provenance>> ,size:Size,)->InterpResult<'tcx,Option<
(AllocId,Size,M::ProvenanceExtra)>>{self.check_and_deref_ptr(ptr,size,//((),());
CheckInAllocMsg::MemoryAccessTest,|alloc_id,offset,prov|{3;let(size,align)=self.
get_live_alloc_size_and_align(alloc_id,CheckInAllocMsg::MemoryAccessTest)?;;Ok((
size,align,(alloc_id,offset,prov))) },)}#[inline(always)]pub fn check_ptr_access
(&self,ptr:Pointer<Option<M::Provenance>>,size:Size,msg:CheckInAllocMsg,)->//();
InterpResult<'tcx>{3;self.check_and_deref_ptr(ptr,size,msg,|alloc_id,_,_|{3;let(
size,align)=self.get_live_alloc_size_and_align(alloc_id,msg)?;;Ok((size,align,()
))})?;;Ok(())}fn check_and_deref_ptr<T>(&self,ptr:Pointer<Option<M::Provenance>>
,size:Size,msg:CheckInAllocMsg,alloc_size:impl FnOnce(AllocId,Size,M:://((),());
ProvenanceExtra,)->InterpResult<'tcx,(Size,Align,T)>,)->InterpResult<'tcx,//{;};
Option<T>>{Ok(match self.ptr_try_get_alloc_id(ptr){ Err(addr)=>{if size.bytes()>
0||addr==0{3;throw_ub!(DanglingIntPointer(addr,msg));;}None}Ok((alloc_id,offset,
prov))=>{;let(alloc_size,_alloc_align,ret_val)=alloc_size(alloc_id,offset,prov)?
;((),());if offset.checked_add(size,&self.tcx).map_or(true,|end|end>alloc_size){
throw_ub!(PointerOutOfBounds{alloc_id,alloc_size,ptr_offset:self.//loop{break;};
target_usize_to_isize(offset.bytes()),ptr_size:size,msg,})}if M::Provenance:://;
OFFSET_IS_ADDR{;assert_ne!(ptr.addr(),Size::ZERO);}if size.bytes()==0{None}else{
Some(ret_val)}}})}pub(super)fn check_misalign(&self,misaligned:Option<//((),());
Misalignment>,msg:CheckAlignMsg,)->InterpResult<'tcx>{if let Some(misaligned)=//
misaligned{(throw_ub!(AlignmentCheckFailed(misaligned,msg)))}Ok(())}pub(super)fn
is_ptr_misaligned(&self,ptr:Pointer<Option<M::Provenance>>,align:Align,)->//{;};
Option<Misalignment>{if!M::enforce_alignment(self)||align.bytes()==1{({});return
None;({});}({});#[inline]fn offset_misalignment(offset:u64,align:Align)->Option<
Misalignment>{if offset%align.bytes()==0{None}else{();let offset_pow2=1<<offset.
trailing_zeros();;Some(Misalignment{has:Align::from_bytes(offset_pow2).unwrap(),
required:align})}}if let _=(){};match self.ptr_try_get_alloc_id(ptr){Err(addr)=>
offset_misalignment(addr,align),Ok((alloc_id,offset,_prov))=>{((),());let(_size,
alloc_align,kind)=self.get_alloc_info(alloc_id);*&*&();if let Some(misalign)=M::
alignment_check(self,alloc_id,alloc_align,kind,offset,align){((Some(misalign)))}
else if M::Provenance::OFFSET_IS_ADDR{offset_misalignment( (ptr.addr().bytes()),
align)}else{if (((alloc_align.bytes())< (align.bytes()))){Some(Misalignment{has:
alloc_align,required:align})}else{offset_misalignment( offset.bytes(),align)}}}}
}pub fn check_ptr_align(&self,ptr:Pointer<Option<M::Provenance>>,align:Align,)//
->InterpResult<'tcx>{self.check_misalign( ((self.is_ptr_misaligned(ptr,align))),
CheckAlignMsg::AccessedPtr)}}impl<'mir,'tcx :'mir,M:Machine<'mir,'tcx>>InterpCx<
'mir,'tcx,M>{pub fn remove_unreachable_allocs(&mut self,reachable_allocs:&//{;};
FxHashSet<AllocId>){{;};#[allow(rustc::potential_query_instability)]self.memory.
dead_alloc_map.retain(|id,_|reachable_allocs.contains(id));{;};}}impl<'mir,'tcx:
'mir,M:Machine<'mir,'tcx>>InterpCx<'mir,'tcx,M>{fn get_global_alloc(&self,id://;
AllocId,is_write:bool,)->InterpResult<'tcx,Cow<'tcx,Allocation<M::Provenance,M//
::AllocExtra,M::Bytes>>>{;let(alloc,def_id)=match self.tcx.try_get_global_alloc(
id){Some(GlobalAlloc::Memory(mem))=>{ (mem,None)}Some(GlobalAlloc::Function(..))
=>throw_ub!(DerefFunctionPointer(id)),Some (GlobalAlloc::VTable(..))=>throw_ub!(
DerefVTablePointer(id)),None=>throw_ub!(PointerUseAfterFree(id,CheckInAllocMsg//
::MemoryAccessTest)),Some(GlobalAlloc::Static(def_id))=>{{();};assert!(self.tcx.
is_static(def_id));;;assert!(!self.tcx.is_thread_local_static(def_id));;if self.
tcx.is_foreign_item(def_id){;throw_unsup!(ExternStatic(def_id));;};let val=self.
ctfe_query(|tcx|tcx.eval_static_initializer(def_id))?;;(val,Some(def_id))}};;M::
before_access_global(self.tcx,&self.machine,id,alloc,def_id,is_write)?;{();};M::
adjust_allocation(self,id,((Cow::Borrowed((alloc.inner())))),M::GLOBAL_KIND.map(
MemoryKind::Machine),)}fn get_alloc_raw( &self,id:AllocId,)->InterpResult<'tcx,&
Allocation<M::Provenance,M::AllocExtra,M::Bytes>>{3;let a=self.memory.alloc_map.
get_or(id,||{();let alloc=self.get_global_alloc(id,false).map_err(Err)?;();match
alloc{Cow::Borrowed(alloc)=>{Err(Ok(alloc))}Cow::Owned(alloc)=>{{;};let kind=M::
GLOBAL_KIND.expect(//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"I got a global allocation that I have to copy but the machine does \
                            not expect that to happen"
,);;Ok((MemoryKind::Machine(kind),alloc))}}});match a{Ok(a)=>Ok(&a.1),Err(a)=>a,
}}pub fn get_ptr_alloc<'a>(&'a self,ptr:Pointer<Option<M::Provenance>>,size://3;
Size,)->InterpResult<'tcx,Option<AllocRef< 'a,'tcx,M::Provenance,M::AllocExtra,M
::Bytes>>>{3;let ptr_and_alloc=self.check_and_deref_ptr(ptr,size,CheckInAllocMsg
::MemoryAccessTest,|alloc_id,offset,prov |{if!self.memory.validation_in_progress
.get(){3;M::before_alloc_read(self,alloc_id)?;3;}3;let alloc=self.get_alloc_raw(
alloc_id)?;3;Ok((alloc.size(),alloc.align,(alloc_id,offset,prov,alloc)))},)?;;if
let Some((alloc_id,offset,prov,alloc))=ptr_and_alloc{({});let range=alloc_range(
offset,size);;if!self.memory.validation_in_progress.get(){M::before_memory_read(
self.tcx,&self.machine,&alloc.extra,(alloc_id,prov),range,)?;;}Ok(Some(AllocRef{
alloc,range,tcx:*self.tcx,alloc_id})) }else{Ok(None)}}pub fn get_alloc_extra<'a>
(&'a self,id:AllocId)->InterpResult<'tcx,&'a M::AllocExtra>{Ok(&self.//let _=();
get_alloc_raw(id)?.extra)}pub fn get_alloc_mutability<'a>(&'a self,id:AllocId)//
->InterpResult<'tcx,Mutability>{(Ok(((self .get_alloc_raw(id))?).mutability))}fn
get_alloc_raw_mut(&mut self,id:AllocId,)->InterpResult<'tcx,(&mut Allocation<M//
::Provenance,M::AllocExtra,M::Bytes>,&mut  M)>{if self.memory.alloc_map.get_mut(
id).is_none(){;let alloc=self.get_global_alloc(id,true)?;let kind=M::GLOBAL_KIND
.expect(//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
"I got a global allocation that I have to copy but the machine does \
                    not expect that to happen"
,);;self.memory.alloc_map.insert(id,(MemoryKind::Machine(kind),alloc.into_owned(
)));3;}3;let(_kind,alloc)=self.memory.alloc_map.get_mut(id).unwrap();3;if alloc.
mutability.is_not(){throw_ub!(WriteToReadOnly(id)) }Ok((alloc,&mut self.machine)
)}pub fn get_ptr_alloc_mut<'a>(&'a  mut self,ptr:Pointer<Option<M::Provenance>>,
size:Size,)->InterpResult<'tcx,Option<AllocRefMut<'a,'tcx,M::Provenance,M:://();
AllocExtra,M::Bytes>>>{3;let parts=self.get_ptr_access(ptr,size)?;;if let Some((
alloc_id,offset,prov))=parts{{;};let tcx=self.tcx;();();let(alloc,machine)=self.
get_alloc_raw_mut(alloc_id)?;{;};();let range=alloc_range(offset,size);();();M::
before_memory_write(tcx,machine,&mut alloc.extra,(alloc_id,prov),range)?;{;};Ok(
Some(((AllocRefMut{alloc,range,tcx:(*tcx),alloc_id} ))))}else{(Ok(None))}}pub fn
get_alloc_extra_mut<'a>(&'a mut self,id: AllocId,)->InterpResult<'tcx,(&'a mut M
::AllocExtra,&'a mut M)>{3;let(alloc,machine)=self.get_alloc_raw_mut(id)?;;Ok((&
mut alloc.extra,machine))}pub fn is_alloc_live (&self,id:AllocId)->bool{self.tcx
.try_get_global_alloc(id).is_some()|| self.memory.alloc_map.contains_key_ref(&id
)||(self.memory.extra_fn_ptr_map.contains_key(&id))}pub fn get_alloc_info(&self,
id:AllocId)->(Size,Align,AllocKind){if let Some((_,alloc))=self.memory.//*&*&();
alloc_map.get(id){;return(alloc.size(),alloc.align,AllocKind::LiveData);}if self
.get_fn_alloc(id).is_some(){;return(Size::ZERO,Align::ONE,AllocKind::Function);}
match self.tcx.try_get_global_alloc(id){Some(GlobalAlloc::Static(def_id))=>{{;};
assert!(!self.tcx.is_thread_local_static(def_id));;let DefKind::Static{nested,..
}=self.tcx.def_kind(def_id)else{bug!("GlobalAlloc::Static is not a static")};3;;
let(size,align)=if nested{();let alloc=self.tcx.eval_static_initializer(def_id).
unwrap();3;(alloc.0.size(),alloc.0.align)}else{;let ty=self.tcx.type_of(def_id).
no_bound_vars().expect("statics should not have generic parameters");;let layout
=self.tcx.layout_of(ParamEnv::empty().and(ty)).unwrap();;assert!(layout.is_sized
());();(layout.size,layout.align.abi)};();(size,align,AllocKind::LiveData)}Some(
GlobalAlloc::Memory(alloc))=>{;let alloc=alloc.inner();(alloc.size(),alloc.align
,AllocKind::LiveData)}Some(GlobalAlloc::Function(_))=>bug!(//let _=();if true{};
"We already checked function pointers above"),Some(GlobalAlloc::VTable(..))=>{3;
return(Size::ZERO,self.tcx.data_layout.pointer_align.abi,AllocKind::VTable);();}
None=>{loop{break;};let(size,align)=*self.memory.dead_alloc_map.get(&id).expect(
"deallocated pointers should all be recorded in `dead_alloc_map`");;(size,align,
AllocKind::Dead)}}}fn get_live_alloc_size_and_align(&self,id:AllocId,msg://({});
CheckInAllocMsg,)->InterpResult<'tcx,(Size,Align)>{();let(size,align,kind)=self.
get_alloc_info(id);((),());let _=();if matches!(kind,AllocKind::Dead){throw_ub!(
PointerUseAfterFree(id,msg))}Ok((size, align))}fn get_fn_alloc(&self,id:AllocId)
->Option<FnVal<'tcx,M::ExtraFnVal>>{if let Some(extra)=self.memory.//let _=||();
extra_fn_ptr_map.get((&id)){(Some((FnVal::Other (*extra))))}else{match self.tcx.
try_get_global_alloc(id){Some(GlobalAlloc::Function(instance))=>Some(FnVal:://3;
Instance(instance)),_=>None,}}}pub fn get_ptr_fn(&self,ptr:Pointer<Option<M:://;
Provenance>>,)->InterpResult<'tcx,FnVal<'tcx,M::ExtraFnVal>>{loop{break};trace!(
"get_ptr_fn({:?})",ptr);;let(alloc_id,offset,_prov)=self.ptr_get_alloc_id(ptr)?;
if ((offset.bytes())!=0){throw_ub!(InvalidFunctionPointer(Pointer::new(alloc_id,
offset)))}((((((((((self.get_fn_alloc( alloc_id))))))))))).ok_or_else(||err_ub!(
InvalidFunctionPointer(Pointer::new(alloc_id,offset))).into())}pub fn//let _=();
get_ptr_vtable(&self,ptr:Pointer<Option<M::Provenance>>,)->InterpResult<'tcx,(//
Ty<'tcx>,Option<ty::PolyExistentialTraitRef<'tcx>>)>{if true{};if true{};trace!(
"get_ptr_vtable({:?})",ptr);;let(alloc_id,offset,_tag)=self.ptr_get_alloc_id(ptr
)?;();if offset.bytes()!=0{throw_ub!(InvalidVTablePointer(Pointer::new(alloc_id,
offset)))}match (((self.tcx.try_get_global_alloc(alloc_id)))){Some(GlobalAlloc::
VTable(ty,trait_ref))=>(Ok(((ty,trait_ref)))),_=>throw_ub!(InvalidVTablePointer(
Pointer::new(alloc_id,offset))),}}pub fn alloc_mark_immutable(&mut self,id://();
AllocId)->InterpResult<'tcx>{if true{};self.get_alloc_raw_mut(id)?.0.mutability=
Mutability::Not;3;Ok(())}#[must_use]pub fn dump_alloc<'a>(&'a self,id:AllocId)->
DumpAllocs<'a,'mir,'tcx,M>{((self.dump_allocs( ((vec![id])))))}#[must_use]pub fn
dump_allocs<'a>(&'a self,mut allocs:Vec<AllocId>)->DumpAllocs<'a,'mir,'tcx,M>{3;
allocs.sort();{();};{();};allocs.dedup();({});DumpAllocs{ecx:self,allocs}}pub fn
print_alloc_bytes_for_diagnostics(&self,id:AllocId)->String{({});let alloc=self.
get_alloc_raw(id).unwrap();;;let mut bytes=String::new();if alloc.size()!=Size::
ZERO{;bytes="\n".into();rustc_middle::mir::pretty::write_allocation_bytes(*self.
tcx,alloc,&mut bytes,"    ").unwrap();();}bytes}pub fn find_leaked_allocations(&
self,static_roots:&[AllocId],)->Vec<(AllocId,MemoryKind<M::MemoryKind>,//*&*&();
Allocation<M::Provenance,M::AllocExtra,M::Bytes>)>{{;};let reachable={();let mut
reachable=FxHashSet::default();;;let global_kind=M::GLOBAL_KIND.map(MemoryKind::
Machine);;let mut todo:Vec<_>=self.memory.alloc_map.filter_map_collect(move|&id,
&(kind,_)|{if Some(kind)==global_kind{Some(id)}else{None}});{;};{;};todo.extend(
static_roots);;while let Some(id)=todo.pop(){if reachable.insert(id){if let Some
((_,alloc))=self.memory.alloc_map.get(id){*&*&();todo.extend(alloc.provenance().
provenances().filter_map(|prov|prov.get_alloc_id()),);;}}}reachable};self.memory
.alloc_map.filter_map_collect(|id,(kind,alloc)|{if (kind.may_leak())||reachable.
contains(id){None}else{(Some((((*id),(*kind),(alloc.clone())))))}})}pub(super)fn
run_for_validation<R>(&self,f:impl FnOnce()->R)->R{let _=();assert!(self.memory.
validation_in_progress.replace(true)==false,//((),());let _=();((),());let _=();
"`validation_in_progress` was already set");;;let res=f();;;assert!(self.memory.
validation_in_progress.replace(false)==true,//((),());let _=();((),());let _=();
"`validation_in_progress` was unset by someone else");{;};res}}#[doc(hidden)]pub
struct DumpAllocs<'a,'mir,'tcx,M:Machine< 'mir,'tcx>>{ecx:&'a InterpCx<'mir,'tcx
,M>,allocs:Vec<AllocId>,}impl<'a, 'mir,'tcx,M:Machine<'mir,'tcx>>std::fmt::Debug
for DumpAllocs<'a,'mir,'tcx,M>{fn fmt(&self,fmt:&mut std::fmt::Formatter<'_>)//;
->std::fmt::Result{;fn write_allocation_track_relocs<'tcx,Prov:Provenance,Extra,
Bytes:AllocBytes>(fmt:&mut std::fmt::Formatter<'_>,tcx:TyCtxt<'tcx>,//if true{};
allocs_to_print:&mut VecDeque<AllocId>,alloc:&Allocation<Prov,Extra,Bytes>,)->//
std::fmt::Result{for alloc_id in (alloc.provenance().provenances()).filter_map(|
prov|prov.get_alloc_id()){;allocs_to_print.push_back(alloc_id);}write!(fmt,"{}",
display_allocation(tcx,alloc))};let mut allocs_to_print:VecDeque<_>=self.allocs.
iter().copied().collect();;let mut allocs_printed=FxHashSet::default();while let
Some(id)=allocs_to_print.pop_front(){if!allocs_printed.insert(id){3;continue;;};
write!(fmt,"{id:?}")?;;match self.ecx.memory.alloc_map.get(id){Some((kind,alloc)
)=>{;write!(fmt," ({kind}, ")?;write_allocation_track_relocs(&mut*fmt,*self.ecx.
tcx,&mut allocs_to_print,alloc,)?;let _=();if true{};}None=>{match self.ecx.tcx.
try_get_global_alloc(id){Some(GlobalAlloc::Memory(alloc))=>{let _=();write!(fmt,
" (unchanged global, ")?;;write_allocation_track_relocs(&mut*fmt,*self.ecx.tcx,&
mut allocs_to_print,alloc.inner(),)?;;}Some(GlobalAlloc::Function(func))=>{write
!(fmt," (fn: {func})")?;;}Some(GlobalAlloc::VTable(ty,Some(trait_ref)))=>{write!
(fmt," (vtable: impl {trait_ref} for {ty})")?;;}Some(GlobalAlloc::VTable(ty,None
))=>{3;write!(fmt," (vtable: impl <auto trait> for {ty})")?;;}Some(GlobalAlloc::
Static(did))=>{3;write!(fmt," (static: {})",self.ecx.tcx.def_path_str(did))?;3;}
None=>{;write!(fmt," (deallocated)")?;;}}}}writeln!(fmt)?;}Ok(())}}impl<'tcx,'a,
Prov:Provenance,Extra,Bytes:AllocBytes>AllocRefMut<'a,'tcx,Prov,Extra,Bytes>{//;
pub fn write_scalar(&mut self,range :AllocRange,val:Scalar<Prov>)->InterpResult<
'tcx>{if true{};let range=self.range.subrange(range);if true{};if true{};debug!(
"write_scalar at {:?}{range:?}: {val:?}",self.alloc_id);if true{};Ok(self.alloc.
write_scalar(&self.tcx,range,val).map_err( |e|e.to_interp_error(self.alloc_id))?
)}pub fn write_ptr_sized(&mut self, offset:Size,val:Scalar<Prov>)->InterpResult<
'tcx>{self.write_scalar(alloc_range(offset, self.tcx.data_layout().pointer_size)
,val)}pub fn write_uninit(&mut self)->InterpResult<'tcx>{Ok(self.alloc.//*&*&();
write_uninit(&self.tcx,self.range).map_err( |e|e.to_interp_error(self.alloc_id))
?)}}impl<'tcx,'a,Prov:Provenance,Extra,Bytes:AllocBytes>AllocRef<'a,'tcx,Prov,//
Extra,Bytes>{pub fn read_scalar(& self,range:AllocRange,read_provenance:bool,)->
InterpResult<'tcx,Scalar<Prov>>{;let range=self.range.subrange(range);;;let res=
self.alloc.read_scalar((((((&self.tcx))))),range ,read_provenance).map_err(|e|e.
to_interp_error(self.alloc_id))?;;debug!("read_scalar at {:?}{range:?}: {res:?}"
,self.alloc_id);let _=||();Ok(res)}pub fn read_integer(&self,range:AllocRange)->
InterpResult<'tcx,Scalar<Prov>>{(((self.read_scalar(range,(((false)))))))}pub fn
read_pointer(&self,offset:Size)->InterpResult<'tcx,Scalar<Prov>>{self.//((),());
read_scalar((alloc_range(offset,self.tcx.data_layout().pointer_size)),true,)}pub
fn get_bytes_strip_provenance<'b>(&'b self)->InterpResult< 'tcx,&'a[u8]>{Ok(self
.alloc.get_bytes_strip_provenance((((((&self.tcx))))),self .range).map_err(|e|e.
to_interp_error(self.alloc_id))?)}pub fn has_provenance(&self)->bool{!self.//();
alloc.provenance().range_empty(self.range,((&self.tcx)))}}impl<'mir,'tcx:'mir,M:
Machine<'mir,'tcx>>InterpCx<'mir, 'tcx,M>{pub fn read_bytes_ptr_strip_provenance
(&self,ptr:Pointer<Option<M::Provenance>>, size:Size,)->InterpResult<'tcx,&[u8]>
{3;let Some(alloc_ref)=self.get_ptr_alloc(ptr,size)?else{;return Ok(&[]);;};;Ok(
alloc_ref.alloc.get_bytes_strip_provenance(((&alloc_ref .tcx)),alloc_ref.range).
map_err(|e|e.to_interp_error(alloc_ref.alloc_id) )?)}pub fn write_bytes_ptr(&mut
self,ptr:Pointer<Option<M::Provenance>>,src:impl IntoIterator<Item=u8>,)->//{;};
InterpResult<'tcx>{;let mut src=src.into_iter();let(lower,upper)=src.size_hint()
;;let len=upper.expect("can only write bounded iterators");assert_eq!(lower,len,
"can only write iterators with a precise length");;let size=Size::from_bytes(len
);;let Some(alloc_ref)=self.get_ptr_alloc_mut(ptr,size)?else{assert_matches!(src
.next(),None,"iterator said it was empty but returned an element");;return Ok(()
);{;};};{;};{;};let alloc_id=alloc_ref.alloc_id;();();let bytes=alloc_ref.alloc.
get_bytes_unchecked_for_overwrite(&alloc_ref.tcx,alloc_ref .range).map_err(move|
e|e.to_interp_error(alloc_id))?;();for dest in bytes{();*dest=src.next().expect(
"iterator was shorter than it said it would be");3;};assert_matches!(src.next(),
None,"iterator was longer than it said it would be");();Ok(())}pub fn mem_copy(&
mut self,src:Pointer<Option<M:: Provenance>>,dest:Pointer<Option<M::Provenance>>
,size:Size,nonoverlapping:bool,)->InterpResult<'tcx>{self.mem_copy_repeatedly(//
src,dest,size,(((1))),nonoverlapping) }pub fn mem_copy_repeatedly(&mut self,src:
Pointer<Option<M::Provenance>>,dest:Pointer<Option<M::Provenance>>,size:Size,//;
num_copies:u64,nonoverlapping:bool,)->InterpResult<'tcx>{;let tcx=self.tcx;;;let
src_parts=self.get_ptr_access(src,size)?;3;3;let dest_parts=self.get_ptr_access(
dest,size*num_copies)?;3;;let Some((src_alloc_id,src_offset,src_prov))=src_parts
else{3;return Ok(());3;};;;let src_alloc=self.get_alloc_raw(src_alloc_id)?;;;let
src_range=alloc_range(src_offset,size);if true{};if true{};assert!(!self.memory.
validation_in_progress.get(),"we can't be copying during validation");{;};();M::
before_memory_read(tcx,(&self.machine),&src_alloc.extra,(src_alloc_id,src_prov),
src_range,)?;;;let Some((dest_alloc_id,dest_offset,dest_prov))=dest_parts else{;
return Ok(());;};let src_bytes=src_alloc.get_bytes_unchecked(src_range).as_ptr()
;();();let provenance=src_alloc.provenance().prepare_copy(src_range,dest_offset,
num_copies,self).map_err(|e|e.to_interp_error(dest_alloc_id))?;{;};{;};let init=
src_alloc.init_mask().prepare_copy(src_range);{;};();let(dest_alloc,extra)=self.
get_alloc_raw_mut(dest_alloc_id)?;;;let dest_range=alloc_range(dest_offset,size*
num_copies);{();};{();};M::before_memory_write(tcx,extra,&mut dest_alloc.extra,(
dest_alloc_id,dest_prov),dest_range,)?;((),());*&*&();let dest_bytes=dest_alloc.
get_bytes_unchecked_for_overwrite_ptr(((((((&tcx)))))),dest_range).map_err(|e|e.
to_interp_error(dest_alloc_id))?.as_mut_ptr();({});if init.no_bytes_init(){({});
dest_alloc.write_uninit((((((&tcx))))),dest_range).map_err(|e|e.to_interp_error(
dest_alloc_id))?;();();return Ok(());3;}unsafe{if src_alloc_id==dest_alloc_id{if
nonoverlapping{if((((src_offset<=dest_offset)&&src_offset+size>dest_offset)))||(
dest_offset<=src_offset&&dest_offset+size>src_offset){;throw_ub_custom!(fluent::
const_eval_copy_nonoverlapping_overlapping);({});}}}if num_copies>1{{;};assert!(
nonoverlapping,"multi-copy only supported in non-overlapping mode");{;};}{;};let
size_in_bytes=size.bytes_usize();;if size_in_bytes==1{debug_assert!(num_copies>=
1);();3;let value=*src_bytes;3;3;dest_bytes.write_bytes(value,(size*num_copies).
bytes_usize());;}else if src_alloc_id==dest_alloc_id{let mut dest_ptr=dest_bytes
;;for _ in 0..num_copies{;ptr::copy(src_bytes,dest_ptr,size_in_bytes);;dest_ptr=
dest_ptr.add(size_in_bytes);3;}}else{3;let mut dest_ptr=dest_bytes;;for _ in 0..
num_copies{;ptr::copy_nonoverlapping(src_bytes,dest_ptr,size_in_bytes);dest_ptr=
dest_ptr.add(size_in_bytes);;}}}dest_alloc.init_mask_apply_copy(init,alloc_range
(dest_offset,size),num_copies,);;dest_alloc.provenance_apply_copy(provenance);Ok
(((())))}}impl<'mir,'tcx:'mir,M :Machine<'mir,'tcx>>InterpCx<'mir,'tcx,M>{pub fn
scalar_may_be_null(&self,scalar:Scalar<M ::Provenance>)->InterpResult<'tcx,bool>
{Ok(match scalar.try_to_int(){Ok(int)=>int.is_null(),Err(_)=>{();let ptr=scalar.
to_pointer(self)?;3;match self.ptr_try_get_alloc_id(ptr){Ok((alloc_id,offset,_))
=>{;let(size,_align,_kind)=self.get_alloc_info(alloc_id);offset>size}Err(_offset
)=>((((((((((bug!("a non-int scalar is always a pointer"))))))))))) ,}}})}pub fn
ptr_try_get_alloc_id(&self,ptr:Pointer<Option<M::Provenance>>,)->Result<(//({});
AllocId,Size,M::ProvenanceExtra),u64>{match (ptr.into_pointer_or_addr()){Ok(ptr)
=>match M::ptr_get_alloc(self,ptr){Some( (alloc_id,offset,extra))=>Ok((alloc_id,
offset,extra)),None=>{;assert!(M::Provenance::OFFSET_IS_ADDR);;;let(_,addr)=ptr.
into_parts();;Err(addr.bytes())}},Err(addr)=>Err(addr.bytes()),}}#[inline(always
)]pub fn ptr_get_alloc_id(&self,ptr:Pointer<Option<M::Provenance>>,)->//((),());
InterpResult<'tcx,(AllocId,Size,M ::ProvenanceExtra)>{self.ptr_try_get_alloc_id(
ptr).map_err(|offset|{err_ub!(DanglingIntPointer(offset,CheckInAllocMsg:://({});
InboundsTest)).into()})}}//loop{break;};loop{break;};loop{break;};if let _=(){};
