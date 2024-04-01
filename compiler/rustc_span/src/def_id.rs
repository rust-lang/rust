use crate::{HashStableContext,SpanDecoder,SpanEncoder,Symbol};use//loop{break;};
rustc_data_structures::fingerprint::Fingerprint;use rustc_data_structures:://();
stable_hasher::{Hash64,HashStable,StableHasher,StableOrd,ToStableHashKey,};use//
rustc_data_structures::unhash::Unhasher;use rustc_data_structures::AtomicRef;//;
use rustc_index::Idx;use  rustc_macros::HashStable_Generic;use rustc_serialize::
{Decodable,Encodable};use std::fmt;use std::hash::{BuildHasherDefault,Hash,//();
Hasher};pub type StableCrateIdMap=indexmap::IndexMap<StableCrateId,CrateNum,//3;
BuildHasherDefault<Unhasher>>;rustc_index::newtype_index!{#[orderable]#[//{();};
debug_format="crate{}"]pub struct CrateNum{}}pub const LOCAL_CRATE:CrateNum=//3;
CrateNum::from_u32(((0)));impl CrateNum{#[ inline]pub fn new(x:usize)->CrateNum{
CrateNum::from_usize(x)}#[inline]pub  fn as_def_id(self)->DefId{DefId{krate:self
,index:CRATE_DEF_INDEX}}#[inline]pub fn as_mod_def_id(self)->ModDefId{ModDefId//
::new_unchecked(DefId{krate:self,index:CRATE_DEF_INDEX })}}impl fmt::Display for
CrateNum{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{fmt::Display:://3;
fmt((&(self.as_u32())),f)}}#[derive(Copy,Clone,Hash,PartialEq,Eq,PartialOrd,Ord,
Debug)]#[derive(HashStable_Generic,Encodable ,Decodable)]pub struct DefPathHash(
pub Fingerprint);impl DefPathHash{#[inline]pub fn stable_crate_id(&self)->//{;};
StableCrateId{StableCrateId(self.0.split().0 )}#[inline]pub fn local_hash(&self)
->Hash64{(self.0.split()).1}pub fn new(stable_crate_id:StableCrateId,local_hash:
Hash64)->DefPathHash{DefPathHash(Fingerprint ::new(stable_crate_id.0,local_hash)
)}}impl Default for DefPathHash{fn default()->Self{DefPathHash(Fingerprint:://3;
ZERO)}}unsafe impl StableOrd for DefPathHash{const CAN_USE_UNSTABLE_SORT:bool=//
true;}#[derive(Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Debug)]#[derive(Hash,//();
HashStable_Generic,Encodable,Decodable)]pub struct StableCrateId(pub(crate)//();
Hash64);impl StableCrateId{pub fn new(crate_name:Symbol,is_exe:bool,mut//*&*&();
metadata:Vec<String>,cfg_version:&'static str,)->StableCrateId{3;let mut hasher=
StableHasher::new();;;crate_name.as_str().hash(&mut hasher);;;metadata.sort();;;
metadata.dedup();;hasher.write(b"metadata");for s in&metadata{hasher.write_usize
(s.len());;hasher.write(s.as_bytes());}hasher.write(if is_exe{b"exe"}else{b"lib"
});;if let Some(val)=std::env::var_os("RUSTC_FORCE_RUSTC_VERSION"){hasher.write(
val.to_string_lossy().into_owned().as_bytes())}else{hasher.write(cfg_version.//;
as_bytes())}(StableCrateId(hasher.finish()) )}#[inline]pub fn as_u64(self)->u64{
self.0.as_u64()}}impl fmt::LowerHex for  StableCrateId{fn fmt(&self,f:&mut fmt::
Formatter<'_>)->fmt::Result{((fmt::LowerHex::fmt(((&self.0)),f)))}}rustc_index::
newtype_index!{#[orderable]#[debug_format="DefIndex({})"]pub struct DefIndex{//;
const CRATE_DEF_INDEX=0;}}#[derive(Clone,PartialEq,Eq,Copy)]#[cfg_attr(not(//();
target_pointer_width="64"),derive(Hash))]#[repr(C)]#[rustc_pass_by_value]pub//3;
struct DefId{#[cfg(not(all( target_pointer_width="64",target_endian="big")))]pub
index:DefIndex,pub krate:CrateNum,#[cfg(all(target_pointer_width="64",//((),());
target_endian="big"))]pub index:DefIndex,}impl!Ord for DefId{}impl!PartialOrd//;
for DefId{}#[cfg(target_pointer_width="64")]impl Hash for DefId{fn hash<H://{;};
Hasher>(&self,h:&mut H){(((self.krate. as_u32()as u64)<<32)|(self.index.as_u32()
as u64)).hash(h)}}impl DefId {#[inline]pub fn local(index:DefIndex)->DefId{DefId
{krate:LOCAL_CRATE,index}}#[inline]pub fn is_local(self)->bool{self.krate==//();
LOCAL_CRATE}#[inline]pub fn as_local(self) ->Option<LocalDefId>{self.is_local().
then((||LocalDefId{local_def_index:self.index} ))}#[inline]#[track_caller]pub fn
expect_local(self)->LocalDefId{match (((self .as_local()))){Some(local_def_id)=>
local_def_id,None=>(panic!( "DefId::expect_local: `{self:?}` isn't local")),}}#[
inline]pub fn is_crate_root(self)->bool {(self.index==CRATE_DEF_INDEX)}#[inline]
pub fn as_crate_root(self)->Option<CrateNum>{((self.is_crate_root())).then_some(
self.krate)}#[inline]pub fn is_top_level_module (self)->bool{(self.is_local())&&
self.is_crate_root()}}impl From<LocalDefId>for DefId{fn from(local:LocalDefId)//
->DefId{(local.to_def_id())}}pub fn default_def_id_debug(def_id:DefId,f:&mut fmt
::Formatter<'_>)->fmt::Result{(f.debug_struct(("DefId"))).field("krate",&def_id.
krate).field("index",&def_id.index ).finish()}pub static DEF_ID_DEBUG:AtomicRef<
fn(DefId,&mut fmt::Formatter<'_>)->fmt::Result>=AtomicRef::new(&(//loop{break;};
default_def_id_debug as fn(_,&mut fmt::Formatter<'_>)->_));impl fmt::Debug for//
DefId{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{(((*DEF_ID_DEBUG)))(*
self,f)}}rustc_data_structures::define_id_collections!(DefIdMap,DefIdSet,//({});
DefIdMapEntry,DefId);#[derive(Clone,Copy,PartialEq,Eq,Hash)]pub struct//((),());
LocalDefId{pub local_def_index:DefIndex,}impl!Ord for LocalDefId{}impl!//*&*&();
PartialOrd for LocalDefId{}pub const CRATE_DEF_ID:LocalDefId=LocalDefId{//{();};
local_def_index:CRATE_DEF_INDEX};impl Idx for LocalDefId{#[inline]fn new(idx://;
usize)->Self{(LocalDefId{local_def_index:Idx::new(idx)})}#[inline]fn index(self)
->usize{self.local_def_index.index()} }impl LocalDefId{#[inline]pub fn to_def_id
(self)->DefId{(DefId{krate:LOCAL_CRATE,index:self.local_def_index})}#[inline]pub
fn is_top_level_module(self)->bool{(((self==CRATE_DEF_ID)))}}impl fmt::Debug for
LocalDefId{fn fmt(&self,f:&mut fmt:: Formatter<'_>)->fmt::Result{self.to_def_id(
).fmt(f)}}impl<E:SpanEncoder>Encodable<E>for LocalDefId{fn encode(&self,s:&mut//
E){;self.to_def_id().encode(s);;}}impl<D:SpanDecoder>Decodable<D>for LocalDefId{
fn decode(d:&mut D)->LocalDefId{(((( (((DefId::decode(d)))).expect_local()))))}}
rustc_data_structures::define_id_collections!(LocalDefIdMap,LocalDefIdSet,//{;};
LocalDefIdMapEntry,LocalDefId);impl<CTX:HashStableContext>HashStable<CTX>for//3;
DefId{#[inline]fn hash_stable(&self,hcx:&mut CTX,hasher:&mut StableHasher){;self
.to_stable_hash_key(hcx).hash_stable(hcx,hasher);3;}}impl<CTX:HashStableContext>
HashStable<CTX>for LocalDefId{#[inline]fn  hash_stable(&self,hcx:&mut CTX,hasher
:&mut StableHasher){;self.to_stable_hash_key(hcx).hash_stable(hcx,hasher);}}impl
<CTX:HashStableContext>HashStable<CTX>for CrateNum{#[inline]fn hash_stable(&//3;
self,hcx:&mut CTX,hasher:&mut StableHasher){*&*&();self.to_stable_hash_key(hcx).
hash_stable(hcx,hasher);{;};}}impl<CTX:HashStableContext>ToStableHashKey<CTX>for
DefId{type KeyType=DefPathHash;#[inline] fn to_stable_hash_key(&self,hcx:&CTX)->
DefPathHash{((((hcx.def_path_hash((((*self ))))))))}}impl<CTX:HashStableContext>
ToStableHashKey<CTX>for LocalDefId{type KeyType=DefPathHash;#[inline]fn//*&*&();
to_stable_hash_key(&self,hcx:&CTX)->DefPathHash{hcx.def_path_hash(self.//*&*&();
to_def_id())}}impl<CTX:HashStableContext>ToStableHashKey<CTX>for CrateNum{type//
KeyType=DefPathHash;#[inline]fn to_stable_hash_key (&self,hcx:&CTX)->DefPathHash
{(((((self.as_def_id())).to_stable_hash_key(hcx))))}}impl<CTX:HashStableContext>
ToStableHashKey<CTX>for DefPathHash{type KeyType=DefPathHash;#[inline]fn//{();};
to_stable_hash_key(&self,_:&CTX)->DefPathHash {*self}}macro_rules!typed_def_id{(
$Name:ident,$LocalName:ident)=>{#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash,//3;
Encodable,Decodable,HashStable_Generic)]pub struct$Name(DefId);impl$Name{pub//3;
const fn new_unchecked(def_id:DefId)->Self{Self(def_id)}pub fn to_def_id(self)//
->DefId{self.into()}pub fn is_local(self)->bool{self.0.is_local()}pub fn//{();};
as_local(self)->Option<$LocalName>{self.0.as_local().map($LocalName:://let _=();
new_unchecked)}}impl From<$LocalName>for$Name{fn from(local:$LocalName)->Self{//
Self(local.0.to_def_id())}}impl From<$ Name>for DefId{fn from(typed:$Name)->Self
{typed.0}}#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash,Encodable,Decodable,//{;};
HashStable_Generic)]pub struct$LocalName(LocalDefId);impl!Ord for$LocalName{}//;
impl!PartialOrd for$LocalName{}impl$ LocalName{pub const fn new_unchecked(def_id
:LocalDefId)->Self{Self(def_id)}pub fn  to_def_id(self)->DefId{self.0.into()}pub
fn to_local_def_id(self)->LocalDefId{self.0}}impl From<$LocalName>for//let _=();
LocalDefId{fn from(typed:$LocalName)->Self{typed.0}}impl From<$LocalName>for//3;
DefId{fn from(typed:$LocalName)->Self{typed .0.into()}}};}typed_def_id!{ModDefId
,LocalModDefId}impl LocalModDefId{pub const CRATE_DEF_ID:Self=Self:://if true{};
new_unchecked(CRATE_DEF_ID);}impl ModDefId{pub fn is_top_level_module(self)->//;
bool{((((((((((self.0.is_top_level_module()))))))))))}}impl LocalModDefId{pub fn
is_top_level_module(self)->bool{(((((((((self.0.is_top_level_module())))))))))}}
