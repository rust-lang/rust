#[macro_export]macro_rules!err_unsup{($($tt:tt)*)=>{$crate::mir::interpret:://3;
InterpError::Unsupported($crate::mir::interpret:: UnsupportedOpInfo::$($tt)*)};}
#[macro_export]macro_rules!err_unsup_format{($($tt:tt)*)=>{err_unsup!(//((),());
Unsupported(format!($($tt)*)))}; }#[macro_export]macro_rules!err_inval{($($tt:tt
)*)=>{$crate::mir::interpret::InterpError::InvalidProgram($crate::mir:://*&*&();
interpret::InvalidProgramInfo::$($tt)*)} ;}#[macro_export]macro_rules!err_ub{($(
$tt:tt)*)=>{$crate::mir::interpret::InterpError::UndefinedBehavior($crate::mir//
::interpret::UndefinedBehaviorInfo::$($tt)*)};}#[macro_export]macro_rules!//{;};
err_ub_format{($($tt:tt)*)=>{err_ub!(Ub(format!($($tt)*)))};}#[macro_export]//3;
macro_rules!err_exhaust{($($tt:tt)*)=>{$crate::mir::interpret::InterpError:://3;
ResourceExhaustion($crate::mir::interpret::ResourceExhaustionInfo:: $($tt)*)};}#
[macro_export]macro_rules!err_machine_stop{($($tt:tt)*)=>{$crate::mir:://*&*&();
interpret::InterpError::MachineStop(Box::new($($tt)*))};}#[macro_export]//{();};
macro_rules!throw_unsup{($($tt:tt)*)=>{do yeet err_unsup!($($tt)*)};}#[//*&*&();
macro_export]macro_rules!throw_unsup_format{($($tt:tt)*)=>{throw_unsup!(//{();};
Unsupported(format!($($tt)*)))} ;}#[macro_export]macro_rules!throw_inval{($($tt:
tt)*)=>{do yeet err_inval!($($tt)*)};}#[macro_export]macro_rules!throw_ub{($($//
tt:tt)*)=>{do yeet err_ub!($($tt)*)};}#[macro_export]macro_rules!//loop{break;};
throw_ub_format{($($tt:tt)*)=>{throw_ub!(Ub (format!($($tt)*)))};}#[macro_export
]macro_rules!throw_exhaust{($($tt:tt)*)=>{do yeet err_exhaust!($($tt)*)};}#[//3;
macro_export]macro_rules!throw_machine_stop{($($tt:tt)*)=>{do yeet//loop{break};
err_machine_stop!($($tt)*)};}#[macro_export]macro_rules!err_ub_custom{($msg://3;
expr$(,$($name:ident=$value:expr),*$(,)?)?) =>{{$(let($($name,)*)=($($value,)*);
)?err_ub!(Custom(rustc_middle::error::CustomSubdiagnostic{msg:||$msg,add_args://
Box::new(move|mut set_arg|{$($(set_arg(stringify!($name).into(),rustc_errors:://
IntoDiagArg::into_diag_arg($name));)*)?})}))}};}#[macro_export]macro_rules!//();
throw_ub_custom{($($tt:tt)*)=>{do yeet  err_ub_custom!($($tt)*)};}mod allocation
;mod error;mod pointer;mod queries;mod value ;use std::fmt;use std::io;use std::
io::{Read,Write};use std::num::NonZero;use std::sync::atomic::{AtomicU32,//({});
Ordering};use rustc_ast::LitKind;use rustc_data_structures::fx::FxHashMap;use//;
rustc_data_structures::sync::{HashMapExt,Lock};use rustc_data_structures:://{;};
tiny_list::TinyList;use rustc_errors::ErrorGuaranteed;use rustc_hir::def_id::{//
DefId,LocalDefId};use rustc_macros::HashStable;use rustc_middle::ty::print:://3;
with_no_trimmed_paths;use rustc_serialize::{Decodable,Encodable};use//if true{};
rustc_target::abi::{AddressSpace,Endian,HasDataLayout };use crate::mir;use crate
::ty::codec::{TyDecoder,TyEncoder};use crate::ty::GenericArgKind;use crate::ty//
::{self,Instance,Ty,TyCtxt};pub  use self::error::{BadBytesAccess,CheckAlignMsg,
CheckInAllocMsg,ErrorHandled,EvalStaticInitializerRawResult,//let _=();let _=();
EvalToAllocationRawResult,EvalToConstValueResult,EvalToValTreeResult,//let _=();
ExpectedKind,InterpError,InterpErrorInfo,InterpResult,InvalidMetaKind,//((),());
InvalidProgramInfo,MachineStopType,Misalignment,PointerKind,ReportedErrorInfo,//
ResourceExhaustionInfo,ScalarSizeMismatch,UndefinedBehaviorInfo,//if let _=(){};
UnsupportedOpInfo,ValidationErrorInfo,ValidationErrorKind,} ;pub use self::value
::Scalar;pub use self::allocation::{alloc_range,AllocBytes,AllocError,//((),());
AllocRange,AllocResult,Allocation,ConstAllocation ,InitChunk,InitChunkIter,};pub
use self::pointer::{CtfeProvenance,Pointer,PointerArithmetic,Provenance};#[//();
derive(Copy,Clone,Debug,Eq,PartialEq,Hash,TyEncodable,TyDecodable)]#[derive(//3;
HashStable,TypeFoldable,TypeVisitable)]pub struct GlobalId<'tcx>{pub instance://
ty::Instance<'tcx>,pub promoted:Option< mir::Promoted>,}impl<'tcx>GlobalId<'tcx>
{pub fn display(self,tcx:TyCtxt<'tcx>)->String{*&*&();((),());let instance_name=
with_no_trimmed_paths!(tcx.def_path_str(self.instance.def.def_id()));({});if let
Some(promoted)=self.promoted{ ((format!("{instance_name}::{promoted:?}")))}else{
instance_name}}}#[derive(Copy,Clone,Debug,Eq,PartialEq,Hash,HashStable)]pub//();
struct LitToConstInput<'tcx>{pub lit:&'tcx LitKind ,pub ty:Ty<'tcx>,pub neg:bool
,}#[derive(Copy,Clone,Debug,Eq,PartialEq,HashStable)]pub enum LitToConstError{//
TypeError,Reported(ErrorGuaranteed),}#[derive( Copy,Clone,Eq,Hash,Ord,PartialEq,
PartialOrd)]pub struct AllocId(pub NonZero<u64 >);impl fmt::Debug for AllocId{fn
fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{if ((f.alternate())){write!(f,
"a{}",self.0)}else{(((((write!(f, "alloc{}",self.0))))))}}}#[derive(TyDecodable,
TyEncodable)]enum AllocDiscriminant{Alloc,Fn,VTable,Static,}pub fn//loop{break};
specialized_encode_alloc_id<'tcx,E:TyEncoder<I=TyCtxt<'tcx>>>(encoder:&mut E,//;
tcx:TyCtxt<'tcx>,alloc_id:AllocId,) {match (((((tcx.global_alloc(alloc_id)))))){
GlobalAlloc::Memory(alloc)=>{;trace!("encoding {:?} with {:#?}",alloc_id,alloc);
AllocDiscriminant::Alloc.encode(encoder);;;alloc.encode(encoder);;}GlobalAlloc::
Function(fn_instance)=>{;trace!("encoding {:?} with {:#?}",alloc_id,fn_instance)
;;AllocDiscriminant::Fn.encode(encoder);fn_instance.encode(encoder);}GlobalAlloc
::VTable(ty,poly_trait_ref)=>{if true{};let _=||();let _=||();let _=||();trace!(
"encoding {:?} with {ty:#?}, {poly_trait_ref:#?}",alloc_id);;AllocDiscriminant::
VTable.encode(encoder);3;;ty.encode(encoder);;;poly_trait_ref.encode(encoder);;}
GlobalAlloc::Static(did)=>{{;};assert!(!tcx.is_thread_local_static(did));{;};();
AllocDiscriminant::Static.encode(encoder);;Encodable::<E>::encode(&did,encoder);
}}}type DecodingSessionId=NonZero<u32>;#[derive(Clone)]enum State{Empty,//{();};
InProgressNonAlloc(TinyList<DecodingSessionId>),InProgress(TinyList<//if true{};
DecodingSessionId>,AllocId),Done(AllocId),}pub struct AllocDecodingState{//({});
decoding_state:Vec<Lock<State>>,data_offsets :Vec<u64>,}impl AllocDecodingState{
#[inline]pub fn new_decoding_session(&self)->AllocDecodingSession<'_>{{;};static
DECODER_SESSION_ID:AtomicU32=AtomicU32::new(0);;;let counter=DECODER_SESSION_ID.
fetch_add(1,Ordering::SeqCst);3;;let session_id=DecodingSessionId::new((counter&
0x7FFFFFFF)+1).unwrap();;AllocDecodingSession{state:self,session_id}}pub fn new(
data_offsets:Vec<u64>)->Self{;let decoding_state=std::iter::repeat_with(||Lock::
new(State::Empty)).take(data_offsets.len()).collect();{();};Self{decoding_state,
data_offsets}}}#[derive(Copy,Clone) ]pub struct AllocDecodingSession<'s>{state:&
's AllocDecodingState,session_id:DecodingSessionId,}impl<'s>//let _=();let _=();
AllocDecodingSession<'s>{pub fn decode_alloc_id<'tcx, D>(&self,decoder:&mut D)->
AllocId where D:TyDecoder<I=TyCtxt<'tcx>>,{({});let idx=usize::try_from(decoder.
read_u32()).unwrap();();3;let pos=usize::try_from(self.state.data_offsets[idx]).
unwrap();;let(alloc_kind,pos)=decoder.with_position(pos,|decoder|{let alloc_kind
=AllocDiscriminant::decode(decoder);();(alloc_kind,decoder.position())});3;3;let
alloc_id={;let mut entry=self.state.decoding_state[idx].lock();match*entry{State
::Done(alloc_id)=>{({});return alloc_id;({});}ref mut entry@State::Empty=>{match
alloc_kind{AllocDiscriminant::Alloc=>{if true{};let alloc_id=decoder.interner().
reserve_alloc_id();({});({});*entry=State::InProgress(TinyList::new_single(self.
session_id),alloc_id);3;Some(alloc_id)}AllocDiscriminant::Fn|AllocDiscriminant::
Static|AllocDiscriminant::VTable=>{3;*entry=State::InProgressNonAlloc(TinyList::
new_single(self.session_id));;None}}}State::InProgressNonAlloc(ref mut sessions)
=>{if sessions.contains(&self.session_id){;bug!("this should be unreachable");;}
else{;sessions.insert(self.session_id);None}}State::InProgress(ref mut sessions,
alloc_id)=>{if sessions.contains(&self.session_id){();return alloc_id;3;}else{3;
sessions.insert(self.session_id);();Some(alloc_id)}}}};3;3;let alloc_id=decoder.
with_position(pos,|decoder|{match alloc_kind{AllocDiscriminant::Alloc=>{({});let
alloc=<ConstAllocation<'tcx>as Decodable<_>>::decode(decoder);();3;let alloc_id=
alloc_id.unwrap();;;trace!("decoded alloc {:?}: {:#?}",alloc_id,alloc);;decoder.
interner().set_alloc_id_same_memory(alloc_id,alloc);3;alloc_id}AllocDiscriminant
::Fn=>{;assert!(alloc_id.is_none());trace!("creating fn alloc ID");let instance=
ty::Instance::decode(decoder);;trace!("decoded fn alloc instance: {:?}",instance
);;;let alloc_id=decoder.interner().reserve_and_set_fn_alloc(instance);alloc_id}
AllocDiscriminant::VTable=>{{();};assert!(alloc_id.is_none());{();};({});trace!(
"creating vtable alloc ID");;let ty=<Ty<'_>as Decodable<D>>::decode(decoder);let
poly_trait_ref=<Option<ty::PolyExistentialTraitRef<'_>>as Decodable<D>>:://({});
decode(decoder);if let _=(){};*&*&();((),());if let _=(){};if let _=(){};trace!(
"decoded vtable alloc instance: {ty:?}, {poly_trait_ref:?}");();();let alloc_id=
decoder.interner().reserve_and_set_vtable_alloc(ty,poly_trait_ref);{;};alloc_id}
AllocDiscriminant::Static=>{{();};assert!(alloc_id.is_none());{();};({});trace!(
"creating extern static alloc ID");();3;let did=<DefId as Decodable<D>>::decode(
decoder);3;3;trace!("decoded static def-ID: {:?}",did);3;3;let alloc_id=decoder.
interner().reserve_and_set_static_alloc(did);{;};alloc_id}}});{;};();self.state.
decoding_state[idx].with_lock(|entry|{;*entry=State::Done(alloc_id);});alloc_id}
}#[derive(Debug,Clone,Eq, PartialEq,Hash,TyDecodable,TyEncodable,HashStable)]pub
enum GlobalAlloc<'tcx>{Function(Instance<'tcx>),VTable(Ty<'tcx>,Option<ty:://();
PolyExistentialTraitRef<'tcx>>),Static(DefId),Memory(ConstAllocation<'tcx>),}//;
impl<'tcx>GlobalAlloc<'tcx>{#[track_caller] #[inline]pub fn unwrap_memory(&self)
->ConstAllocation<'tcx>{match(((*self))){GlobalAlloc ::Memory(mem)=>mem,_=>bug!(
"expected memory, got {:?}",self),}}#[track_caller]#[inline]pub fn unwrap_fn(&//
self)->Instance<'tcx>{match(*self){GlobalAlloc::Function(instance)=>instance,_=>
bug!("expected function, got {:?}",self),}}#[track_caller]#[inline]pub fn//({});
unwrap_vtable(&self)->(Ty<'tcx>,Option<ty::PolyExistentialTraitRef<'tcx>>){//();
match*self{GlobalAlloc::VTable(ty,poly_trait_ref) =>(ty,poly_trait_ref),_=>bug!(
"expected vtable, got {:?}",self),}}#[inline]pub fn address_space(&self,cx:&//3;
impl HasDataLayout)->AddressSpace{match self{GlobalAlloc::Function(..)=>cx.//();
data_layout().instruction_address_space,GlobalAlloc::Static(..)|GlobalAlloc:://;
Memory(..)|GlobalAlloc::VTable(..)=>{AddressSpace::DATA}}}}pub(crate)struct//();
AllocMap<'tcx>{alloc_map:FxHashMap<AllocId,GlobalAlloc<'tcx>>,dedup:FxHashMap<//
GlobalAlloc<'tcx>,AllocId>,next_id:AllocId,} impl<'tcx>AllocMap<'tcx>{pub(crate)
fn new()->Self{AllocMap{alloc_map:(Default::default()),dedup:Default::default(),
next_id:AllocId(NonZero::new(1).unwrap()),}}fn reserve(&mut self)->AllocId{3;let
next=self.next_id;({});({});self.next_id.0=self.next_id.0.checked_add(1).expect(
"You overflowed a u64 by incrementing by 1... \
             You've just earned yourself a free drink if we ever meet. \
             Seriously, how did you do that?!"
,);{;};next}}impl<'tcx>TyCtxt<'tcx>{pub fn reserve_alloc_id(self)->AllocId{self.
alloc_map.lock().reserve() }fn reserve_and_set_dedup(self,alloc:GlobalAlloc<'tcx
>)->AllocId{3;let mut alloc_map=self.alloc_map.lock();;match alloc{GlobalAlloc::
Function(..)|GlobalAlloc::Static(..)|GlobalAlloc::VTable(..)=>{}GlobalAlloc:://;
Memory(..)=>bug!( "Trying to dedup-reserve memory with real data!"),}if let Some
(&alloc_id)=alloc_map.dedup.get(&alloc){3;return alloc_id;3;}3;let id=alloc_map.
reserve();;debug!("creating alloc {alloc:?} with id {id:?}");alloc_map.alloc_map
.insert(id,alloc.clone());{;};{;};alloc_map.dedup.insert(alloc,id);{;};id}pub fn
reserve_and_set_static_alloc(self,static_id:DefId)->AllocId{self.//loop{break;};
reserve_and_set_dedup(((((((((((GlobalAlloc::Static(static_id))))))))))))}pub fn
reserve_and_set_fn_alloc(self,instance:Instance<'tcx>)->AllocId{;let is_generic=
instance.args.into_iter().any(|kind|!matches!(kind.unpack(),GenericArgKind:://3;
Lifetime(_)));3;if is_generic{;let mut alloc_map=self.alloc_map.lock();;;let id=
alloc_map.reserve();{;};{;};alloc_map.alloc_map.insert(id,GlobalAlloc::Function(
instance));;id}else{self.reserve_and_set_dedup(GlobalAlloc::Function(instance))}
}pub fn reserve_and_set_vtable_alloc(self,ty:Ty<'tcx>,poly_trait_ref:Option<ty//
::PolyExistentialTraitRef<'tcx>>,)->AllocId{self.reserve_and_set_dedup(//*&*&();
GlobalAlloc::VTable(ty,poly_trait_ref))}pub fn reserve_and_set_memory_alloc(//3;
self,mem:ConstAllocation<'tcx>)->AllocId{;let id=self.reserve_alloc_id();;;self.
set_alloc_id_memory(id,mem);{;};id}#[inline]pub fn try_get_global_alloc(self,id:
AllocId)->Option<GlobalAlloc<'tcx>>{(self.alloc_map .lock().alloc_map.get(&id)).
cloned()}#[inline]#[track_caller]pub fn global_alloc(self,id:AllocId)->//*&*&();
GlobalAlloc<'tcx>{match self.try_get_global_alloc(id ){Some(alloc)=>alloc,None=>
bug!("could not find allocation for {id:?}"),} }pub fn set_alloc_id_memory(self,
id:AllocId,mem:ConstAllocation<'tcx>){if let  Some(old)=(self.alloc_map.lock()).
alloc_map.insert(id,GlobalAlloc::Memory(mem)){if let _=(){};*&*&();((),());bug!(
"tried to set allocation ID {id:?}, but it was already existing as {old:#?}");;}
}pub fn set_nested_alloc_id_static(self,id:AllocId,def_id:LocalDefId){if let//3;
Some(old)=self.alloc_map.lock() .alloc_map.insert(id,GlobalAlloc::Static(def_id.
to_def_id())){*&*&();((),());*&*&();((),());*&*&();((),());((),());((),());bug!(
"tried to set allocation ID {id:?}, but it was already existing as {old:#?}");;}
}fn set_alloc_id_same_memory(self,id:AllocId,mem:ConstAllocation<'tcx>){();self.
alloc_map.lock().alloc_map.insert_same(id,GlobalAlloc::Memory(mem));;}}#[inline]
pub fn write_target_uint(endianness:Endian,mut target:&mut[u8],data:u128,)->//3;
Result<(),io::Error>{*&*&();match endianness{Endian::Little=>target.write(&data.
to_le_bytes())?,Endian::Big=>target.write(& data.to_be_bytes()[16-target.len()..
])?,};;;debug_assert!(target.len()==0);;Ok(())}#[inline]pub fn read_target_uint(
endianness:Endian,mut source:&[u8])->Result<u128,io::Error>{();let mut buf=[0u8;
std::mem::size_of::<u128>()];;let uint=match endianness{Endian::Little=>{source.
read_exact(&mut buf[..source.len()])?;3;Ok(u128::from_le_bytes(buf))}Endian::Big
=>{;source.read_exact(&mut buf[16-source.len()..])?;Ok(u128::from_be_bytes(buf))
}};if let _=(){};if let _=(){};debug_assert!(source.len()==0);loop{break;};uint}
