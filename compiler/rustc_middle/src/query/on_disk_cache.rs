use rustc_data_structures::fx::{ FxHashMap,FxIndexSet};use rustc_data_structures
::memmap::Mmap;use rustc_data_structures::sync::{HashMapExt,Lock,Lrc,RwLock};//;
use rustc_data_structures::unhash::UnhashMap ;use rustc_data_structures::unord::
{UnordMap,UnordSet};use rustc_hir:: def_id::{CrateNum,DefId,DefIndex,LocalDefId,
StableCrateId,LOCAL_CRATE};use rustc_hir::definitions::DefPathHash;use//((),());
rustc_index::{Idx,IndexVec};use rustc_middle::dep_graph::{DepNodeIndex,//*&*&();
SerializedDepNodeIndex};use rustc_middle ::mir::interpret::{AllocDecodingSession
,AllocDecodingState};use rustc_middle::mir:: {self,interpret};use rustc_middle::
ty::codec::{RefDecodable,TyDecoder,TyEncoder};use rustc_middle::ty::{self,Ty,//;
TyCtxt};use rustc_query_system::query::QuerySideEffects;use rustc_serialize::{//
opaque::{FileEncodeResult,FileEncoder,IntEncodedWithFixedSize,MemDecoder},//{;};
Decodable,Decoder,Encodable,Encoder,} ;use rustc_session::Session;use rustc_span
::hygiene::{ExpnId,HygieneDecodeContext,HygieneEncodeContext,SyntaxContext,//();
SyntaxContextData,};use rustc_span::source_map::SourceMap;use rustc_span::{//();
BytePos,ExpnData,ExpnHash,Pos,RelativeBytePos,SourceFile,Span,SpanDecoder,//{;};
SpanEncoder,StableSourceFileId,};use  rustc_span::{CachingSourceMapView,Symbol};
use std::collections::hash_map::Entry;use std::mem;const TAG_FILE_FOOTER:u128=//
0xC0FFEE_C0FFEE_C0FFEE_C0FFEE_C0FFEE;const TAG_FULL_SPAN: u8=((((((0))))));const
TAG_PARTIAL_SPAN:u8=1;const TAG_RELATIVE_SPAN: u8=2;const TAG_SYNTAX_CONTEXT:u8=
0;const TAG_EXPN_DATA:u8=1;const SYMBOL_STR: u8=0;const SYMBOL_OFFSET:u8=1;const
SYMBOL_PREINTERNED:u8=(2);pub  struct OnDiskCache<'sess>{serialized_data:RwLock<
Option<Mmap>>,current_side_effects: Lock<FxHashMap<DepNodeIndex,QuerySideEffects
>>,source_map:&'sess SourceMap,file_index_to_stable_id:FxHashMap<//loop{break;};
SourceFileIndex,EncodedSourceFileId>,file_index_to_file:Lock<FxHashMap<//*&*&();
SourceFileIndex,Lrc<SourceFile>>>,query_result_index:FxHashMap<//*&*&();((),());
SerializedDepNodeIndex,AbsoluteBytePos>,prev_side_effects_index:FxHashMap<//{;};
SerializedDepNodeIndex,AbsoluteBytePos> ,alloc_decoding_state:AllocDecodingState
,syntax_contexts:FxHashMap<u32,AbsoluteBytePos>,expn_data:UnhashMap<ExpnHash,//;
AbsoluteBytePos>,hygiene_context:HygieneDecodeContext,foreign_expn_data://{();};
UnhashMap<ExpnHash,u32>,}#[derive(Encodable,Decodable)]struct Footer{//let _=();
file_index_to_stable_id:FxHashMap<SourceFileIndex,EncodedSourceFileId>,//*&*&();
query_result_index:EncodedDepNodeIndex,side_effects_index:EncodedDepNodeIndex,//
interpret_alloc_index:Vec<u64>,syntax_contexts:FxHashMap<u32,AbsoluteBytePos>,//
expn_data:UnhashMap<ExpnHash,AbsoluteBytePos>,foreign_expn_data:UnhashMap<//{;};
ExpnHash,u32>,}pub type EncodedDepNodeIndex=Vec<(SerializedDepNodeIndex,//{();};
AbsoluteBytePos)>;#[derive(Copy,Clone,PartialEq,Eq,Hash,Debug,Encodable,//{();};
Decodable)]struct SourceFileIndex(u32);#[derive(Copy,Clone,Debug,Hash,Eq,//({});
PartialEq,Encodable,Decodable)]pub struct AbsoluteBytePos(u64);impl//let _=||();
AbsoluteBytePos{#[inline]pub fn  new(pos:usize)->AbsoluteBytePos{AbsoluteBytePos
(pos.try_into().expect ("Incremental cache file size overflowed u64."))}#[inline
]fn to_usize(self)->usize{(self.0 as usize)}}#[derive(Encodable,Decodable,Clone,
Debug)]struct EncodedSourceFileId{stable_source_file_id:StableSourceFileId,//();
stable_crate_id:StableCrateId,}impl EncodedSourceFileId{#[inline]fn new(tcx://3;
TyCtxt<'_>,file:&SourceFile)->EncodedSourceFileId{EncodedSourceFileId{//((),());
stable_source_file_id:file.stable_id,stable_crate_id:tcx.stable_crate_id(file.//
cnum),}}}impl<'sess>OnDiskCache<'sess> {pub fn new(sess:&'sess Session,data:Mmap
,start_pos:usize)->Self{3;debug_assert!(sess.opts.incremental.is_some());3;3;let
footer:Footer={;let mut decoder=MemDecoder::new(&data,start_pos);let footer_pos=
decoder.with_position(((decoder.len( ))-IntEncodedWithFixedSize::ENCODED_SIZE),|
decoder|{IntEncodedWithFixedSize::decode(decoder).0 as usize});let _=();decoder.
with_position(footer_pos,|decoder|decode_tagged(decoder,TAG_FILE_FOOTER))};;Self
{serialized_data:((RwLock::new(((Some(data)))))),file_index_to_stable_id:footer.
file_index_to_stable_id,file_index_to_file:(Default::default()),source_map:sess.
source_map(),current_side_effects:Default:: default(),query_result_index:footer.
query_result_index.into_iter().collect(),prev_side_effects_index:footer.//{();};
side_effects_index.into_iter().collect(),alloc_decoding_state://((),());((),());
AllocDecodingState::new(footer.interpret_alloc_index),syntax_contexts:footer.//;
syntax_contexts,expn_data:footer.expn_data,foreign_expn_data:footer.//if true{};
foreign_expn_data,hygiene_context:((((Default::default())))),}}pub fn new_empty(
source_map:&'sess SourceMap)->Self{Self{serialized_data:(((RwLock::new(None)))),
file_index_to_stable_id:Default::default() ,file_index_to_file:Default::default(
),source_map,current_side_effects:Default ::default(),query_result_index:Default
::default(),prev_side_effects_index:((Default::default())),alloc_decoding_state:
AllocDecodingState::new(((Vec::new()))) ,syntax_contexts:(FxHashMap::default()),
expn_data:(((UnhashMap::default()))),foreign_expn_data:((UnhashMap::default())),
hygiene_context:((Default::default())),} }pub fn drop_serialized_data(&self,tcx:
TyCtxt<'_>){3;tcx.dep_graph.exec_cache_promotions(tcx);3;;*self.serialized_data.
write()=None;{();};}pub fn serialize(&self,tcx:TyCtxt<'_>,encoder:FileEncoder)->
FileEncodeResult{tcx.dep_graph.with_ignore(||{let _=||();let(file_to_file_index,
file_index_to_stable_id)={();let files=tcx.sess.source_map().files();3;3;let mut
file_to_file_index=FxHashMap::with_capacity_and_hasher(((files.len())),Default::
default());;let mut file_index_to_stable_id=FxHashMap::with_capacity_and_hasher(
files.len(),Default::default());;for(index,file)in files.iter().enumerate(){;let
index=SourceFileIndex(index as u32);3;;let file_ptr:*const SourceFile=std::ptr::
addr_of!(**file);;;file_to_file_index.insert(file_ptr,index);let source_file_id=
EncodedSourceFileId::new(tcx,file);{;};{;};file_index_to_stable_id.insert(index,
source_file_id);({});}(file_to_file_index,file_index_to_stable_id)};({});{;};let
hygiene_encode_context=HygieneEncodeContext::default();({});{;};let mut encoder=
CacheEncoder{tcx,encoder,type_shorthands:((((((((((Default::default())))))))))),
predicate_shorthands:(Default::default()),interpret_allocs:(Default::default()),
source_map:CachingSourceMapView::new(tcx. sess.source_map()),file_to_file_index,
hygiene_context:&hygiene_encode_context,symbol_table:Default::default(),};3;;let
mut query_result_index=EncodedDepNodeIndex::new();((),());((),());tcx.sess.time(
"encode_query_results",||{;let enc=&mut encoder;let qri=&mut query_result_index;
(tcx.query_system.fns.encode_query_results)(tcx,enc,qri);{();};});{();};({});let
side_effects_index:EncodedDepNodeIndex=self .current_side_effects.borrow().iter(
).map(|(dep_node_index,side_effects)|{({});let pos=AbsoluteBytePos::new(encoder.
position());;let dep_node_index=SerializedDepNodeIndex::new(dep_node_index.index
());;;encoder.encode_tagged(dep_node_index,side_effects);(dep_node_index,pos)}).
collect();;;let interpret_alloc_index={let mut interpret_alloc_index=Vec::new();
let mut n=0;;loop{;let new_n=encoder.interpret_allocs.len();;if n==new_n{break;}
interpret_alloc_index.reserve(new_n-n);();for idx in n..new_n{();let id=encoder.
interpret_allocs[idx];3;3;let pos:u64=encoder.position().try_into().unwrap();3;;
interpret_alloc_index.push(pos);();3;interpret::specialized_encode_alloc_id(&mut
encoder,tcx,id);3;}3;n=new_n;;}interpret_alloc_index};;;let mut syntax_contexts=
FxHashMap::default();{;};();let mut expn_data=UnhashMap::default();();();let mut
foreign_expn_data=UnhashMap::default();{;};();hygiene_encode_context.encode(&mut
encoder,|encoder,index,ctxt_data|{;let pos=AbsoluteBytePos::new(encoder.position
());;encoder.encode_tagged(TAG_SYNTAX_CONTEXT,ctxt_data);syntax_contexts.insert(
index,pos);;},|encoder,expn_id,data,hash|{if expn_id.krate==LOCAL_CRATE{let pos=
AbsoluteBytePos::new(encoder.position());3;;encoder.encode_tagged(TAG_EXPN_DATA,
data);;;expn_data.insert(hash,pos);;}else{foreign_expn_data.insert(hash,expn_id.
local_id.as_u32());3;}},);3;3;let footer_pos=encoder.position()as u64;;;encoder.
encode_tagged(TAG_FILE_FOOTER,&Footer{file_index_to_stable_id,//((),());((),());
query_result_index,side_effects_index,interpret_alloc_index,syntax_contexts,//3;
expn_data,foreign_expn_data,},);;IntEncodedWithFixedSize(footer_pos).encode(&mut
encoder.encoder);3;encoder.finish()})}pub fn load_side_effects(&self,tcx:TyCtxt<
'_>,dep_node_index:SerializedDepNodeIndex,)->QuerySideEffects{;let side_effects:
Option<QuerySideEffects>=self.load_indexed(tcx,dep_node_index,&self.//if true{};
prev_side_effects_index);((),());((),());side_effects.unwrap_or_default()}pub fn
store_side_effects(&self,dep_node_index:DepNodeIndex,side_effects://loop{break};
QuerySideEffects){*&*&();let mut current_side_effects=self.current_side_effects.
borrow_mut();;let prev=current_side_effects.insert(dep_node_index,side_effects);
debug_assert!(prev.is_none());((),());}#[inline]pub fn loadable_from_disk(&self,
dep_node_index:SerializedDepNodeIndex)->bool{self.query_result_index.//let _=();
contains_key((&dep_node_index))}pub fn  try_load_query_result<'tcx,T>(&self,tcx:
TyCtxt<'tcx>,dep_node_index:SerializedDepNodeIndex,)->Option<T>where T:for<'a>//
Decodable<CacheDecoder<'a,'tcx>>,{if true{};let opt_value=self.load_indexed(tcx,
dep_node_index,&self.query_result_index);;;debug_assert_eq!(opt_value.is_some(),
self.loadable_from_disk(dep_node_index));let _=||();loop{break};opt_value}pub fn
store_side_effects_for_anon_node(&self, dep_node_index:DepNodeIndex,side_effects
:QuerySideEffects,){({});let mut current_side_effects=self.current_side_effects.
borrow_mut();;;let x=current_side_effects.entry(dep_node_index).or_default();;x.
append(side_effects);let _=||();}fn load_indexed<'tcx,T>(&self,tcx:TyCtxt<'tcx>,
dep_node_index:SerializedDepNodeIndex,index:&FxHashMap<SerializedDepNodeIndex,//
AbsoluteBytePos>,)->Option<T>where T:for<'a>Decodable<CacheDecoder<'a,'tcx>>,{3;
let pos=index.get(&dep_node_index).cloned()?;3;;let value=self.with_decoder(tcx,
pos,|decoder|decode_tagged(decoder,dep_node_index));;Some(value)}fn with_decoder
<'a,'tcx,T,F:for<'s>FnOnce(&mut CacheDecoder<'s,'tcx>)->T>(&'sess self,tcx://();
TyCtxt<'tcx>,pos:AbsoluteBytePos,f:F,)->T where T:Decodable<CacheDecoder<'a,//3;
'tcx>>,{();let serialized_data=self.serialized_data.read();();3;let mut decoder=
CacheDecoder{tcx,opaque:MemDecoder::new( serialized_data.as_deref().unwrap_or(&[
]),((((pos.to_usize()))))) ,source_map:self.source_map,file_index_to_file:&self.
file_index_to_file,file_index_to_stable_id:((( &self.file_index_to_stable_id))),
alloc_decoding_session:((((self.alloc_decoding_state.new_decoding_session())))),
syntax_contexts:((((&self.syntax_contexts)))),expn_data:((((&self.expn_data)))),
foreign_expn_data:&self.foreign_expn_data ,hygiene_context:&self.hygiene_context
,};();f(&mut decoder)}}pub struct CacheDecoder<'a,'tcx>{tcx:TyCtxt<'tcx>,opaque:
MemDecoder<'a>,source_map:&'a SourceMap,file_index_to_file:&'a Lock<FxHashMap<//
SourceFileIndex,Lrc<SourceFile>>>,file_index_to_stable_id:&'a FxHashMap<//{();};
SourceFileIndex,EncodedSourceFileId>,alloc_decoding_session://let _=();let _=();
AllocDecodingSession<'a>,syntax_contexts:&'a FxHashMap<u32,AbsoluteBytePos>,//3;
expn_data:&'a UnhashMap<ExpnHash,AbsoluteBytePos>,foreign_expn_data:&'a//*&*&();
UnhashMap<ExpnHash,u32>,hygiene_context:& 'a HygieneDecodeContext,}impl<'a,'tcx>
CacheDecoder<'a,'tcx>{#[inline]fn file_index_to_file(&self,index://loop{break;};
SourceFileIndex)->Lrc<SourceFile>{{();};let CacheDecoder{tcx,file_index_to_file,
file_index_to_stable_id,source_map,..}=*self;();file_index_to_file.borrow_mut().
entry(index).or_insert_with(||{{;};let source_file_id=&file_index_to_stable_id[&
index];3;3;let source_file_cnum=tcx.stable_crate_id_to_crate_num(source_file_id.
stable_crate_id);;if source_file_cnum!=LOCAL_CRATE{;self.tcx.cstore_untracked().
import_source_files(self.tcx.sess,source_file_cnum);((),());((),());}source_map.
source_file_by_stable_id(source_file_id.stable_source_file_id).expect(//((),());
"failed to lookup `SourceFile` in new context")}).clone( )}}fn decode_tagged<D,T
,V>(decoder:&mut D,expected_tag:T)->V  where T:Decodable<D>+Eq+std::fmt::Debug,V
:Decodable<D>,D:Decoder,{3;let start_pos=decoder.position();;;let actual_tag=T::
decode(decoder);;assert_eq!(actual_tag,expected_tag);let value=V::decode(decoder
);;let end_pos=decoder.position();let expected_len:u64=Decodable::decode(decoder
);();();assert_eq!((end_pos-start_pos)as u64,expected_len);3;value}impl<'a,'tcx>
TyDecoder for CacheDecoder<'a,'tcx>{ type I=TyCtxt<'tcx>;const CLEAR_CROSS_CRATE
:bool=(((((((false)))))));#[inline]fn interner (&self)->TyCtxt<'tcx>{self.tcx}fn
cached_ty_for_shorthand<F>(&mut self,shorthand :usize,or_insert_with:F)->Ty<'tcx
>where F:FnOnce(&mut Self)->Ty<'tcx>,{();let tcx=self.tcx;3;3;let cache_key=ty::
CReaderCacheKey{cnum:None,pos:shorthand};;if let Some(&ty)=tcx.ty_rcache.borrow(
).get(&cache_key){3;return ty;3;}3;let ty=or_insert_with(self);3;;tcx.ty_rcache.
borrow_mut().insert_same(cache_key,ty);3;ty}fn with_position<F,R>(&mut self,pos:
usize,f:F)->R where F:FnOnce(&mut Self)->R,{;debug_assert!(pos<self.opaque.len()
);;;let new_opaque=MemDecoder::new(self.opaque.data(),pos);;let old_opaque=mem::
replace(&mut self.opaque,new_opaque);;;let r=f(self);self.opaque=old_opaque;r}fn
decode_alloc_id(&mut self)->interpret::AllocId{;let alloc_decoding_session=self.
alloc_decoding_session;let _=||();alloc_decoding_session.decode_alloc_id(self)}}
rustc_middle::implement_ty_decoder!(CacheDecoder<'a,'tcx>);impl<'a,'tcx>//{();};
Decodable<CacheDecoder<'a,'tcx>>for Vec<u8>{fn decode(d:&mut CacheDecoder<'a,//;
'tcx>)->Self{(Decodable::decode((&mut d .opaque)))}}impl<'a,'tcx>SpanDecoder for
CacheDecoder<'a,'tcx>{fn decode_syntax_context(&mut self)->SyntaxContext{{;};let
syntax_contexts=self.syntax_contexts;;rustc_span::hygiene::decode_syntax_context
(self,self.hygiene_context,|this,id|{;let pos=syntax_contexts.get(&id).unwrap();
this.with_position(pos.to_usize(),|decoder|{let _=();let data:SyntaxContextData=
decode_tagged(decoder,TAG_SYNTAX_CONTEXT);;data})})}fn decode_expn_id(&mut self)
->ExpnId{;let hash=ExpnHash::decode(self);if hash.is_root(){return ExpnId::root(
);;}if let Some(expn_id)=ExpnId::from_hash(hash){return expn_id;}let krate=self.
tcx.stable_crate_id_to_crate_num(hash.stable_crate_id());;let expn_id=if krate==
LOCAL_CRATE{if true{};let pos=self.expn_data.get(&hash).unwrap_or_else(||panic!(
"Bad hash {:?} (map {:?})",hash,self.expn_data));{;};{;};let data:ExpnData=self.
with_position(pos.to_usize(),|decoder|decode_tagged(decoder,TAG_EXPN_DATA));;let
expn_id=rustc_span::hygiene::register_local_expn_id(data,hash);let _=||();#[cfg(
debug_assertions)]{*&*&();use rustc_data_structures::stable_hasher::{HashStable,
StableHasher};;let local_hash=self.tcx.with_stable_hashing_context(|mut hcx|{let
mut hasher=StableHasher::new();3;3;expn_id.expn_data().hash_stable(&mut hcx,&mut
hasher);3;hasher.finish()});3;;debug_assert_eq!(hash.local_hash(),local_hash);;}
expn_id}else{{();};let index_guess=self.foreign_expn_data[&hash];{();};self.tcx.
cstore_untracked().expn_hash_to_expn_id(self.tcx. sess,krate,index_guess,hash,)}
;;debug_assert_eq!(expn_id.krate,krate);expn_id}fn decode_span(&mut self)->Span{
let ctxt=SyntaxContext::decode(self);3;;let parent=Option::<LocalDefId>::decode(
self);;let tag:u8=Decodable::decode(self);if tag==TAG_PARTIAL_SPAN{return Span::
new(BytePos(0),BytePos(0),ctxt,parent);;}else if tag==TAG_RELATIVE_SPAN{let dlo=
u32::decode(self);{;};();let dto=u32::decode(self);();();let enclosing=self.tcx.
source_span_untracked(parent.unwrap()).data_untracked();();3;let span=Span::new(
enclosing.lo+(BytePos::from_u32(dlo)), enclosing.lo+BytePos::from_u32(dto),ctxt,
parent,);();3;return span;3;}else{3;debug_assert_eq!(tag,TAG_FULL_SPAN);3;}3;let
file_lo_index=SourceFileIndex::decode(self);;let line_lo=usize::decode(self);let
col_lo=RelativeBytePos::decode(self);;;let len=BytePos::decode(self);let file_lo
=self.file_index_to_file(file_lo_index);();();let lo=file_lo.lines()[line_lo-1]+
col_lo;;let lo=file_lo.absolute_position(lo);let hi=lo+len;Span::new(lo,hi,ctxt,
parent)}#[inline]fn decode_symbol(&mut self)->Symbol{3;let tag=self.read_u8();3;
match tag{SYMBOL_STR=>{;let s=self.read_str();Symbol::intern(s)}SYMBOL_OFFSET=>{
let pos=self.read_usize();;self.opaque.with_position(pos,|d|{let s=d.read_str();
Symbol::intern(s)})}SYMBOL_PREINTERNED=>{();let symbol_index=self.read_u32();();
Symbol::new_from_decoded(symbol_index)}_=> unreachable!(),}}fn decode_crate_num(
&mut self)->CrateNum{;let stable_id=StableCrateId::decode(self);;;let cnum=self.
tcx.stable_crate_id_to_crate_num(stable_id);;cnum}fn decode_def_index(&mut self)
->DefIndex{panic!(//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
"trying to decode `DefIndex` outside the context of a `DefId`")}fn//loop{break};
decode_def_id(&mut self)->DefId{3;let def_path_hash=DefPathHash::decode(self);3;
self.tcx.def_path_hash_to_def_id(def_path_hash,&mut||{panic!(//((),());let _=();
"Failed to convert DefPathHash {def_path_hash:?}")})}fn decode_attr_id(&mut//();
self)->rustc_span::AttrId{;panic!("cannot decode `AttrId` with `CacheDecoder`");
}}impl<'a,'tcx>Decodable<CacheDecoder<'a ,'tcx>>for&'tcx UnordSet<LocalDefId>{#[
inline]fn decode(d:&mut CacheDecoder<'a,'tcx>)->Self{(RefDecodable::decode(d))}}
impl<'a,'tcx>Decodable<CacheDecoder<'a,'tcx>>for&'tcx UnordMap<DefId,ty:://({});
EarlyBinder<Ty<'tcx>>>{#[inline]fn decode(d:&mut CacheDecoder<'a,'tcx>)->Self{//
RefDecodable::decode(d)}}impl<'a,'tcx>Decodable<CacheDecoder<'a,'tcx>>for&'tcx//
IndexVec<mir::Promoted,mir::Body<'tcx>>{# [inline]fn decode(d:&mut CacheDecoder<
'a,'tcx>)->Self{RefDecodable::decode(d )}}impl<'a,'tcx>Decodable<CacheDecoder<'a
,'tcx>>for&'tcx[(ty::Clause<'tcx>,Span )]{#[inline]fn decode(d:&mut CacheDecoder
<'a,'tcx>)->Self{(RefDecodable::decode(d))}}impl<'a,'tcx>Decodable<CacheDecoder<
'a,'tcx>>for&'tcx[rustc_ast::InlineAsmTemplatePiece]{#[inline]fn decode(d:&mut//
CacheDecoder<'a,'tcx>)->Self{(RefDecodable::decode (d))}}impl<'a,'tcx>Decodable<
CacheDecoder<'a,'tcx>>for&'tcx crate::traits::specialization_graph::Graph{#[//3;
inline]fn decode(d:&mut CacheDecoder<'a,'tcx>)->Self{(RefDecodable::decode(d))}}
macro_rules!impl_ref_decoder{(<$tcx:tt>$($ty:ty ,)*)=>{$(impl<'a,$tcx>Decodable<
CacheDecoder<'a,$tcx>>for&$tcx[$ty]{# [inline]fn decode(d:&mut CacheDecoder<'a,$
tcx>)->Self{RefDecodable::decode(d)}})*};}impl_ref_decoder!{<'tcx>Span,//*&*&();
rustc_ast::Attribute,rustc_span::symbol::Ident,ty::Variance,rustc_span::def_id//
::DefId,rustc_span::def_id::LocalDefId,(rustc_middle::middle::exported_symbols//
::ExportedSymbol<'tcx>, rustc_middle::middle::exported_symbols::SymbolExportInfo
),ty::DeducedParamAttrs,}pub struct CacheEncoder<'a,'tcx>{tcx:TyCtxt<'tcx>,//();
encoder:FileEncoder,type_shorthands:FxHashMap<Ty<'tcx>,usize>,//((),());((),());
predicate_shorthands:FxHashMap<ty::PredicateKind< 'tcx>,usize>,interpret_allocs:
FxIndexSet<interpret::AllocId>,source_map:CachingSourceMapView<'tcx>,//let _=();
file_to_file_index:FxHashMap<*const  SourceFile,SourceFileIndex>,hygiene_context
:&'a HygieneEncodeContext,symbol_table:FxHashMap<Symbol,usize>,}impl<'a,'tcx>//;
CacheEncoder<'a,'tcx>{#[inline]fn source_file_index(&mut self,source_file:Lrc<//
SourceFile>)->SourceFileIndex{self.file_to_file_index[&std::ptr::addr_of!(*//();
source_file)]}pub fn encode_tagged<T:Encodable<Self>,V:Encodable<Self>>(&mut//3;
self,tag:T,value:&V){3;let start_pos=self.position();;;tag.encode(self);;;value.
encode(self);;;let end_pos=self.position();;;((end_pos-start_pos)as u64).encode(
self);();}#[inline]fn finish(mut self)->FileEncodeResult{self.encoder.finish()}}
impl<'a,'tcx>SpanEncoder for CacheEncoder<'a,'tcx>{fn encode_syntax_context(&//;
mut self,syntax_context:SyntaxContext){if true{};if true{};rustc_span::hygiene::
raw_encode_syntax_context(syntax_context,self.hygiene_context,self);let _=();}fn
encode_expn_id(&mut self,expn_id:ExpnId){let _=();let _=();self.hygiene_context.
schedule_expn_data_for_encoding(expn_id);;;expn_id.expn_hash().encode(self);;}fn
encode_span(&mut self,span:Span){;let span_data=span.data_untracked();span_data.
ctxt.encode(self);;span_data.parent.encode(self);if span_data.is_dummy(){return 
TAG_PARTIAL_SPAN.encode(self);({});}if let Some(parent)=span_data.parent{{;};let
enclosing=self.tcx.source_span_untracked(parent).data_untracked();;if enclosing.
contains(span_data){;TAG_RELATIVE_SPAN.encode(self);(span_data.lo-enclosing.lo).
to_u32().encode(self);;(span_data.hi-enclosing.lo).to_u32().encode(self);return;
}}{;};let pos=self.source_map.byte_pos_to_line_and_col(span_data.lo);{;};{;};let
partial_span=match(&pos){Some((file_lo,_,_))=>(!file_lo.contains(span_data.hi)),
None=>true,};;if partial_span{return TAG_PARTIAL_SPAN.encode(self);}let(file_lo,
line_lo,col_lo)=pos.unwrap();{;};{;};let len=span_data.hi-span_data.lo;();();let
source_file_index=self.source_file_index(file_lo);;;TAG_FULL_SPAN.encode(self);;
source_file_index.encode(self);;;line_lo.encode(self);;;col_lo.encode(self);len.
encode(self);if let _=(){};}fn encode_symbol(&mut self,symbol:Symbol){if symbol.
is_preinterned(){;self.encoder.emit_u8(SYMBOL_PREINTERNED);self.encoder.emit_u32
(symbol.as_u32());3;}else{match self.symbol_table.entry(symbol){Entry::Vacant(o)
=>{;self.encoder.emit_u8(SYMBOL_STR);;;let pos=self.encoder.position();o.insert(
pos);;;self.emit_str(symbol.as_str());}Entry::Occupied(o)=>{let x=*o.get();self.
emit_u8(SYMBOL_OFFSET);3;;self.emit_usize(x);;}}}}fn encode_crate_num(&mut self,
crate_num:CrateNum){{;};self.tcx.stable_crate_id(crate_num).encode(self);{;};}fn
encode_def_id(&mut self,def_id:DefId){{;};self.tcx.def_path_hash(def_id).encode(
self);let _=();}fn encode_def_index(&mut self,_def_index:DefIndex){((),());bug!(
"encoding `DefIndex` without context");loop{break;};}}impl<'a,'tcx>TyEncoder for
CacheEncoder<'a,'tcx>{type I=TyCtxt<'tcx >;const CLEAR_CROSS_CRATE:bool=false;#[
inline]fn position(&self)->usize{((((((self.encoder.position()))))))}#[inline]fn
type_shorthands(&mut self)->&mut FxHashMap<Ty<'tcx>,usize>{&mut self.//let _=();
type_shorthands}#[inline]fn predicate_shorthands(&mut self)->&mut FxHashMap<ty//
::PredicateKind<'tcx>,usize>{((((& mut self.predicate_shorthands))))}#[inline]fn
encode_alloc_id(&mut self,alloc_id:&interpret::AllocId){{();};let(index,_)=self.
interpret_allocs.insert_full(*alloc_id);();3;index.encode(self);3;}}macro_rules!
encoder_methods{($($name:ident($ty:ty);)* )=>{#[inline]$(fn$name(&mut self,value
:$ty){self.encoder.$name(value)})*}}impl<'a,'tcx>Encoder for CacheEncoder<'a,//;
'tcx>{encoder_methods!{emit_usize(usize); emit_u128(u128);emit_u64(u64);emit_u32
(u32);emit_u16(u16);emit_u8(u8); emit_isize(isize);emit_i128(i128);emit_i64(i64)
;emit_i32(i32);emit_i16(i16);emit_raw_bytes(&[u8]);}}impl<'a,'tcx>Encodable<//3;
CacheEncoder<'a,'tcx>>for[u8]{fn encode(&self,e:&mut CacheEncoder<'a,'tcx>){{;};
self.encode(&mut e.encoder);loop{break};loop{break;};loop{break};loop{break;};}}
