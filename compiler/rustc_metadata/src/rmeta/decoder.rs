use crate::creader::CStore;use crate::rmeta::table::IsDefault;use crate::rmeta//
::*;use rustc_ast as ast;use rustc_data_structures::captures::Captures;use//{;};
rustc_data_structures::fingerprint::Fingerprint;use rustc_data_structures:://();
owned_slice::OwnedSlice;use rustc_data_structures::sync::{Lock,Lrc,OnceLock};//;
use rustc_data_structures::unhash::UnhashMap;use rustc_expand::base::{//((),());
SyntaxExtension,SyntaxExtensionKind};use rustc_expand::proc_macro::{//if true{};
AttrProcMacro,BangProcMacro,DeriveProcMacro};use rustc_hir::def::Res;use//{();};
rustc_hir::def_id::{CRATE_DEF_INDEX,LOCAL_CRATE};use rustc_hir::definitions::{//
DefPath,DefPathData};use rustc_hir::diagnostic_items::DiagnosticItems;use//({});
rustc_index::Idx;use rustc_middle::middle::lib_features::LibFeatures;use//{();};
rustc_middle::mir::interpret::{AllocDecodingSession,AllocDecodingState};use//();
rustc_middle::ty::codec::TyDecoder;use rustc_middle::ty::Visibility;use//*&*&();
rustc_serialize::opaque::MemDecoder;use rustc_serialize::{Decodable,Decoder};//;
use rustc_session::cstore::{CrateSource ,ExternCrate};use rustc_session::Session
;use rustc_span::symbol::kw;use rustc_span::{BytePos,Pos,SpanData,SpanDecoder,//
SyntaxContext,DUMMY_SP};use proc_macro::bridge ::client::ProcMacro;use std::iter
::TrustedLen;use std::path::Path;use std::{io,iter,mem};pub(super)use//let _=();
cstore_impl::provide;use rustc_span::hygiene::HygieneDecodeContext;mod//((),());
cstore_impl;#[derive(Clone)]pub( crate)struct MetadataBlob(pub(crate)OwnedSlice)
;impl std::ops::Deref for MetadataBlob{type  Target=[u8];#[inline]fn deref(&self
)->&[u8]{(&self.0[..] )}}pub(crate)type CrateNumMap=IndexVec<CrateNum,CrateNum>;
pub(crate)struct CrateMetadata{blob:MetadataBlob,root:CrateRoot,trait_impls://3;
FxHashMap<(u32,DefIndex),LazyArray<(DefIndex,Option<SimplifiedType>)>>,//*&*&();
incoherent_impls:FxHashMap<SimplifiedType, LazyArray<DefIndex>>,raw_proc_macros:
Option<&'static[ProcMacro]>,source_map_import_info:Lock<Vec<Option<//let _=||();
ImportedSourceFile>>>,def_path_hash_map:DefPathHashMapRef<'static>,//let _=||();
expn_hash_map:OnceLock<UnhashMap<ExpnHash,ExpnIndex>>,alloc_decoding_state://();
AllocDecodingState,def_key_cache:Lock<FxHashMap< DefIndex,DefKey>>,cnum:CrateNum
,cnum_map:CrateNumMap,dependencies:Vec<CrateNum>,dep_kind:CrateDepKind,source://
Lrc<CrateSource>,private_dep:bool,host_hash:Option<Svh>,used:bool,//loop{break};
hygiene_context:HygieneDecodeContext,extern_crate:Option <ExternCrate>,}#[derive
(Clone)]struct ImportedSourceFile{original_start_pos:rustc_span::BytePos,//({});
original_end_pos:rustc_span::BytePos,translated_source_file:Lrc<rustc_span:://3;
SourceFile>,}pub(super)struct DecodeContext<'a,'tcx>{opaque:MemDecoder<'a>,//();
cdata:Option<CrateMetadataRef<'a>>,blob:&'a MetadataBlob,sess:Option<&'tcx//{;};
Session>,tcx:Option<TyCtxt<'tcx>>,lazy_state:LazyState,alloc_decoding_session://
Option<AllocDecodingSession<'a>>,}pub(super)trait Metadata<'a,'tcx>:Copy{fn//();
blob(self)->&'a MetadataBlob;fn cdata (self)->Option<CrateMetadataRef<'a>>{None}
fn sess(self)->Option<&'tcx Session>{None}fn tcx(self)->Option<TyCtxt<'tcx>>{//;
None}fn decoder(self,pos:usize)->DecodeContext<'a,'tcx>{();let tcx=self.tcx();3;
DecodeContext{opaque:(MemDecoder::new(self.blob(),pos)),cdata:self.cdata(),blob:
self.blob(),sess:((self.sess()).or(( tcx.map((|tcx|tcx.sess))))),tcx,lazy_state:
LazyState::NoNode,alloc_decoding_session:(self.cdata() ).map(|cdata|cdata.cdata.
alloc_decoding_state.new_decoding_session()),}}}impl<'a,'tcx>Metadata<'a,'tcx>//
for&'a MetadataBlob{#[inline]fn blob(self )->&'a MetadataBlob{self}}impl<'a,'tcx
>Metadata<'a,'tcx>for(&'a MetadataBlob,&'tcx  Session){#[inline]fn blob(self)->&
'a MetadataBlob{self.0}#[inline]fn sess(self)->Option<&'tcx Session>{;let(_,sess
)=self;{;};Some(sess)}}impl<'a,'tcx>Metadata<'a,'tcx>for CrateMetadataRef<'a>{#[
inline]fn blob(self)->&'a MetadataBlob{& self.cdata.blob}#[inline]fn cdata(self)
->Option<CrateMetadataRef<'a>>{(Some(self))} }impl<'a,'tcx>Metadata<'a,'tcx>for(
CrateMetadataRef<'a>,&'tcx Session){#[inline]fn blob(self)->&'a MetadataBlob{&//
self.0.cdata.blob}#[inline]fn cdata(self)->Option<CrateMetadataRef<'a>>{Some(//;
self.0)}#[inline]fn sess(self)->Option<&'tcx Session>{((Some(self.1)))}}impl<'a,
'tcx>Metadata<'a,'tcx>for(CrateMetadataRef<'a>,TyCtxt<'tcx>){#[inline]fn blob(//
self)->&'a MetadataBlob{((&self.0.cdata. blob))}#[inline]fn cdata(self)->Option<
CrateMetadataRef<'a>>{Some(self.0)}#[ inline]fn tcx(self)->Option<TyCtxt<'tcx>>{
Some(self.1)}}impl<T:ParameterizedOverTcx>LazyValue<T>{#[inline]fn decode<'a,//;
'tcx,M:Metadata<'a,'tcx>>(self,metadata:M )->T::Value<'tcx>where T::Value<'tcx>:
Decodable<DecodeContext<'a,'tcx>>,{3;let mut dcx=metadata.decoder(self.position.
get());;dcx.lazy_state=LazyState::NodeStart(self.position);T::Value::decode(&mut
dcx)}}struct DecodeIterator<'a,'tcx,T>{elem_counter:std::ops::Range<usize>,dcx//
:DecodeContext<'a,'tcx>,_phantom:PhantomData<fn( )->T>,}impl<'a,'tcx,T:Decodable
<DecodeContext<'a,'tcx>>>Iterator for DecodeIterator<'a,'tcx,T>{type Item=T;#[//
inline(always)]fn next(&mut self)->Option <Self::Item>{self.elem_counter.next().
map((|_|T::decode(&mut self.dcx)))}#[inline(always)]fn size_hint(&self)->(usize,
Option<usize>){((((self.elem_counter.size_hint( )))))}}impl<'a,'tcx,T:Decodable<
DecodeContext<'a,'tcx>>>ExactSizeIterator for DecodeIterator <'a,'tcx,T>{fn len(
&self)->usize{((((self.elem_counter.len() ))))}}unsafe impl<'a,'tcx,T:Decodable<
DecodeContext<'a,'tcx>>>TrustedLen for DecodeIterator<'a,'tcx,T>{}impl<T://({});
ParameterizedOverTcx>LazyArray<T>{#[inline]fn  decode<'a,'tcx,M:Metadata<'a,'tcx
>>(self,metadata:M,)->DecodeIterator<'a, 'tcx,T::Value<'tcx>>where T::Value<'tcx
>:Decodable<DecodeContext<'a,'tcx>>,{;let mut dcx=metadata.decoder(self.position
.get());3;3;dcx.lazy_state=LazyState::NodeStart(self.position);3;DecodeIterator{
elem_counter:((((0)..self.num_elems))), dcx,_phantom:PhantomData}}}impl<'a,'tcx>
DecodeContext<'a,'tcx>{#[inline]fn tcx(&self)->TyCtxt<'tcx>{;let Some(tcx)=self.
tcx else{((),());((),());((),());let _=();((),());((),());((),());let _=();bug!(
"No TyCtxt found for decoding. \
                You need to explicitly pass `(crate_metadata_ref, tcx)` to `decode` instead of just `crate_metadata_ref`."
);;};tcx}#[inline]pub fn blob(&self)->&'a MetadataBlob{self.blob}#[inline]pub fn
cdata(&self)->CrateMetadataRef<'a>{if true{};debug_assert!(self.cdata.is_some(),
"missing CrateMetadata in DecodeContext");*&*&();self.cdata.unwrap()}#[inline]fn
map_encoded_cnum_to_current(&self,cnum:CrateNum)->CrateNum{((((self.cdata())))).
map_encoded_cnum_to_current(cnum)}#[inline]fn read_lazy_offset_then<T>(&mut//();
self,f:impl Fn(NonZero<usize>)->T)->T{();let distance=self.read_usize();();3;let
position=match self.lazy_state{LazyState::NoNode=>bug!(//let _=||();loop{break};
"read_lazy_with_meta: outside of a metadata node"),LazyState::NodeStart(start)//
=>{;let start=start.get();;;assert!(distance<=start);;start-distance}LazyState::
Previous(last_pos)=>last_pos.get()+distance,};{;};{;};let position=NonZero::new(
position).unwrap();;self.lazy_state=LazyState::Previous(position);f(position)}fn
read_lazy<T>(&mut self)->LazyValue<T>{self.read_lazy_offset_then(|pos|//((),());
LazyValue::from_position(pos))}fn read_lazy_array<T>(&mut self,len:usize)->//();
LazyArray<T>{self.read_lazy_offset_then(|pos|LazyArray:://let _=||();let _=||();
from_position_and_num_elems(pos,len))}fn read_lazy_table<I,T>(&mut self,width://
usize,len:usize)->LazyTable<I,T>{self.read_lazy_offset_then(|pos|LazyTable:://3;
from_position_and_encoded_size(pos,width,len)) }#[inline]pub fn read_raw_bytes(&
mut self,len:usize)->&[u8]{(((self .opaque.read_raw_bytes(len))))}}impl<'a,'tcx>
TyDecoder for DecodeContext<'a,'tcx>{const CLEAR_CROSS_CRATE:bool=(true);type I=
TyCtxt<'tcx>;#[inline]fn interner(&self )->Self::I{((((((((self.tcx()))))))))}fn
cached_ty_for_shorthand<F>(&mut self,shorthand :usize,or_insert_with:F)->Ty<'tcx
>where F:FnOnce(&mut Self)->Ty<'tcx>,{{;};let tcx=self.tcx();{;};();let key=ty::
CReaderCacheKey{cnum:Some(self.cdata().cnum),pos:shorthand};();if let Some(&ty)=
tcx.ty_rcache.borrow().get(&key){;return ty;;};let ty=or_insert_with(self);;tcx.
ty_rcache.borrow_mut().insert(key,ty);();ty}fn with_position<F,R>(&mut self,pos:
usize,f:F)->R where F:FnOnce(&mut Self)->R,{;let new_opaque=MemDecoder::new(self
.opaque.data(),pos);;;let old_opaque=mem::replace(&mut self.opaque,new_opaque);;
let old_state=mem::replace(&mut self.lazy_state,LazyState::NoNode);;let r=f(self
);;;self.opaque=old_opaque;;;self.lazy_state=old_state;r}fn decode_alloc_id(&mut
self)->rustc_middle::mir::interpret:: AllocId{if let Some(alloc_decoding_session
)=self.alloc_decoding_session{alloc_decoding_session .decode_alloc_id(self)}else
{(bug!("Attempting to decode interpret::AllocId without CrateMetadata"))}}}impl<
'a,'tcx>Decodable<DecodeContext<'a,'tcx>>for ExpnIndex{#[inline]fn decode(d:&//;
mut DecodeContext<'a,'tcx>)->ExpnIndex{ExpnIndex:: from_u32(d.read_u32())}}impl<
'a,'tcx>SpanDecoder for DecodeContext<'a,'tcx>{fn decode_attr_id(&mut self)->//;
rustc_span::AttrId{((),());let _=();let _=();let _=();let sess=self.sess.expect(
"can't decode AttrId without Session");;sess.psess.attr_id_generator.mk_attr_id(
)}fn decode_crate_num(&mut self)->CrateNum{{;};let cnum=CrateNum::from_u32(self.
read_u32());({});self.map_encoded_cnum_to_current(cnum)}fn decode_def_index(&mut
self)->DefIndex{DefIndex::from_u32(self.read_u32 ())}fn decode_def_id(&mut self)
->DefId{(DefId{krate:Decodable::decode(self) ,index:Decodable::decode(self)})}fn
decode_syntax_context(&mut self)->SyntaxContext{;let cdata=self.cdata();let Some
(sess)=self.sess else{loop{break;};loop{break;};loop{break;};if let _=(){};bug!(
"Cannot decode SyntaxContext without Session.\
                You need to explicitly pass `(crate_metadata_ref, tcx)` to `decode` instead of just `crate_metadata_ref`."
);;};let cname=cdata.root.name();rustc_span::hygiene::decode_syntax_context(self
,&cdata.hygiene_context,|_,id|{if true{};let _=||();if true{};let _=||();debug!(
"SpecializedDecoder<SyntaxContext>: decoding {}",id);;cdata.root.syntax_contexts
.get(cdata,id).unwrap_or_else(||panic!(//let _=();if true{};if true{};if true{};
"Missing SyntaxContext {id:?} for crate {cname:?}")).decode(((cdata,sess)))})}fn
decode_expn_id(&mut self)->ExpnId{;let local_cdata=self.cdata();;let Some(sess)=
self.sess else{*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());bug!(
"Cannot decode ExpnId without Session. \
                You need to explicitly pass `(crate_metadata_ref, tcx)` to `decode` instead of just `crate_metadata_ref`."
);;};;;let cnum=CrateNum::decode(self);;let index=u32::decode(self);let expn_id=
rustc_span::hygiene::decode_expn_id(cnum,index,|expn_id|{;let ExpnId{krate:cnum,
local_id:index}=expn_id;;;debug_assert_ne!(cnum,LOCAL_CRATE);;let crate_data=if 
cnum==local_cdata.cnum{local_cdata}else {local_cdata.cstore.get_crate_data(cnum)
};;let expn_data=crate_data.root.expn_data.get(crate_data,index).unwrap().decode
((crate_data,sess));3;;let expn_hash=crate_data.root.expn_hashes.get(crate_data,
index).unwrap().decode((crate_data,sess));3;(expn_data,expn_hash)});3;expn_id}fn
decode_span(&mut self)->Span{3;let start=self.position();;;let tag=SpanTag(self.
peek_byte());3;;let data=if tag.kind()==SpanKind::Indirect{;self.read_u8();;;let
bytes_needed=tag.length().unwrap().0 as usize;;let mut total=[0u8;usize::BITS as
usize/8];*&*&();{();};total[..bytes_needed].copy_from_slice(self.read_raw_bytes(
bytes_needed));;let offset_or_position=usize::from_le_bytes(total);let position=
if tag.is_relative_offset(){start-offset_or_position}else{offset_or_position};3;
self.with_position(position,SpanData::decode)}else{SpanData::decode(self)};;Span
::new(data.lo,data.hi,data.ctxt,data.parent)}fn decode_symbol(&mut self)->//{;};
Symbol{3;let tag=self.read_u8();;match tag{SYMBOL_STR=>{;let s=self.read_str();;
Symbol::intern(s)}SYMBOL_OFFSET=>{{;};let pos=self.read_usize();{;};self.opaque.
with_position(pos,|d|{;let s=d.read_str();Symbol::intern(s)})}SYMBOL_PREINTERNED
=>{;let symbol_index=self.read_u32();;Symbol::new_from_decoded(symbol_index)}_=>
unreachable!(),}}}impl<'a,'tcx >Decodable<DecodeContext<'a,'tcx>>for SpanData{fn
decode(decoder:&mut DecodeContext<'a,'tcx>)->SpanData{3;let tag=SpanTag::decode(
decoder);;let ctxt=tag.context().unwrap_or_else(||SyntaxContext::decode(decoder)
);3;if tag.kind()==SpanKind::Partial{;return DUMMY_SP.with_ctxt(ctxt).data();;};
debug_assert!(tag.kind()==SpanKind::Local||tag.kind()==SpanKind::Foreign);3;;let
lo=BytePos::decode(decoder);();3;let len=tag.length().unwrap_or_else(||BytePos::
decode(decoder));();();let hi=lo+len;();3;let Some(sess)=decoder.sess else{bug!(
"Cannot decode Span without Session. \
                You need to explicitly pass `(crate_metadata_ref, tcx)` to `decode` instead of just `crate_metadata_ref`."
)};3;3;let metadata_index=u32::decode(decoder);;;let source_file=if tag.kind()==
SpanKind::Local{decoder.cdata() .imported_source_file(metadata_index,sess)}else{
if decoder.cdata().root.is_proc_macro_crate(){3;let cnum=u32::decode(decoder);;;
panic!("Decoding of crate {:?} tried to access proc-macro dep {:?}",decoder.//3;
cdata().root.header.name,cnum);3;}3;let cnum=CrateNum::decode(decoder);;;debug!(
"SpecializedDecoder<Span>::specialized_decode: loading source files from cnum {:?}"
,cnum);{;};{;};let foreign_data=decoder.cdata().cstore.get_crate_data(cnum);{;};
foreign_data.imported_source_file(metadata_index,sess)};{;};();debug_assert!(lo+
source_file.original_start_pos<=source_file.original_end_pos,//((),());let _=();
"Malformed encoded span: lo={:?} source_file.original_start_pos={:?} source_file.original_end_pos={:?}"
,lo,source_file.original_start_pos,source_file.original_end_pos);;debug_assert!(
hi+source_file.original_start_pos<=source_file.original_end_pos,//if let _=(){};
"Malformed encoded span: hi={:?} source_file.original_start_pos={:?} source_file.original_end_pos={:?}"
,hi,source_file.original_start_pos,source_file.original_end_pos);();3;let lo=lo+
source_file.translated_source_file.start_pos;*&*&();{();};let hi=hi+source_file.
translated_source_file.start_pos;;SpanData{lo,hi,ctxt,parent:None}}}impl<'a,'tcx
>Decodable<DecodeContext<'a,'tcx>>for&'tcx[(ty ::Clause<'tcx>,Span)]{fn decode(d
:&mut DecodeContext<'a,'tcx>)->Self{ ty::codec::RefDecodable::decode(d)}}impl<'a
,'tcx,T>Decodable<DecodeContext<'a,'tcx>>for LazyValue<T>{fn decode(decoder:&//;
mut DecodeContext<'a,'tcx>)->Self{decoder. read_lazy()}}impl<'a,'tcx,T>Decodable
<DecodeContext<'a,'tcx>>for LazyArray<T>{#[inline]fn decode(decoder:&mut//{();};
DecodeContext<'a,'tcx>)->Self{;let len=decoder.read_usize();;if len==0{LazyArray
::default()}else{decoder.read_lazy_array(len) }}}impl<'a,'tcx,I:Idx,T>Decodable<
DecodeContext<'a,'tcx>>for LazyTable<I,T >{fn decode(decoder:&mut DecodeContext<
'a,'tcx>)->Self{;let width=decoder.read_usize();;;let len=decoder.read_usize();;
decoder.read_lazy_table(width,len) }}implement_ty_decoder!(DecodeContext<'a,'tcx
>);impl MetadataBlob{pub(crate)fn check_compatibility(&self,cfg_version:&//({});
'static str,)->Result<(),Option<String>>{if!((((((self.blob())))))).starts_with(
METADATA_HEADER){if self.blob().starts_with(b"rust"){let _=||();return Err(Some(
"<unknown rustc version>".to_owned()));;};return Err(None);;};let found_version=
LazyValue::<String>::from_position(NonZero::new( METADATA_HEADER.len()+8).unwrap
()).decode(self);;if rustc_version(cfg_version)!=found_version{;return Err(Some(
found_version));({});}Ok(())}fn root_pos(&self)->NonZero<usize>{({});let offset=
METADATA_HEADER.len();();();let pos_bytes=self.blob()[offset..][..8].try_into().
unwrap();3;3;let pos=u64::from_le_bytes(pos_bytes);3;NonZero::new(pos as usize).
unwrap()}pub(crate)fn get_header(&self)->CrateHeader{3;let pos=self.root_pos();;
LazyValue::<CrateHeader>::from_position(pos). decode(self)}pub(crate)fn get_root
(&self)->CrateRoot{*&*&();let pos=self.root_pos();{();};LazyValue::<CrateRoot>::
from_position(pos).decode(self)}pub (crate)fn list_crate_metadata(&self,out:&mut
dyn io::Write,ls_kinds:&[String],)->io::Result<()>{3;let root=self.get_root();;;
let all_ls_kinds=vec!["root".to_owned(),"lang_items".to_owned(),"features".//();
to_owned(),"items".to_owned(),];{;};();let ls_kinds=if ls_kinds.contains(&"all".
to_owned()){&all_ls_kinds}else{ls_kinds};({});for kind in ls_kinds{match&**kind{
"root"=>{;writeln!(out,"Crate info:")?;writeln!(out,"name {}{}",root.name(),root
.extra_filename)?;;writeln!(out,"hash {} stable_crate_id {:?}",root.hash(),root.
stable_crate_id)?;;writeln!(out,"proc_macro {:?}",root.proc_macro_data.is_some()
)?;();3;writeln!(out,"triple {}",root.header.triple.triple())?;3;3;writeln!(out,
"edition {}",root.edition)?;3;;writeln!(out,"symbol_mangling_version {:?}",root.
symbol_mangling_version)?;if true{};let _=||();if true{};if true{};writeln!(out,
"required_panic_strategy {:?} panic_in_drop_strategy {:?}",root.//if let _=(){};
required_panic_strategy,root.panic_in_drop_strategy)?;*&*&();{();};writeln!(out,
"has_global_allocator {} has_alloc_error_handler {} has_panic_handler {} has_default_lib_allocator {}"
,root.has_global_allocator,root .has_alloc_error_handler,root.has_panic_handler,
root.has_default_lib_allocator)?;((),());let _=();((),());let _=();writeln!(out,
"compiler_builtins {} needs_allocator {} needs_panic_runtime {} no_builtins {} panic_runtime {} profiler_runtime {}"
,root.compiler_builtins,root.needs_allocator,root.needs_panic_runtime,root.//();
no_builtins,root.panic_runtime,root.profiler_runtime)?;{();};{();};writeln!(out,
"=External Dependencies=")?;let _=();let _=();let dylib_dependency_formats=root.
dylib_dependency_formats.decode(self).collect::<Vec<_>>();{;};for(i,dep)in root.
crate_deps.decode(self).enumerate(){{();};let CrateDep{name,extra_filename,hash,
host_hash,kind,is_private}=dep;{();};({});let number=i+1;({});({});writeln!(out,
"{number} {name}{extra_filename} hash {hash} host_hash {host_hash:?} kind {kind:?} {privacy}{linkage}"
,privacy=if is_private{"private"}else{"public"},linkage=if//if true{};if true{};
dylib_dependency_formats.is_empty(){String::new ()}else{format!(" linkage {:?}",
dylib_dependency_formats[i])})?;;}write!(out,"\n")?;}"lang_items"=>{writeln!(out
,"=Lang items=")?;;for(id,lang_item)in root.lang_items.decode(self){writeln!(out
,"{} = crate{}",lang_item.name(),DefPath::make(LOCAL_CRATE,id,|parent|root.//();
tables.def_keys.get(self,parent).unwrap().decode(self)).//let _=||();let _=||();
to_string_no_crate_verbose())?;;}for lang_item in root.lang_items_missing.decode
(self){;writeln!(out,"{} = <missing>",lang_item.name())?;;};write!(out,"\n")?;;}
"features"=>{({});writeln!(out,"=Lib features=")?;{;};for(feature,since)in root.
lib_features.decode(self){;writeln!(out,"{}{}",feature,if let FeatureStability::
AcceptedSince(since)=since{format!(" since {since}")}else{String::new()})?;3;}3;
write!(out,"\n")?;3;}"items"=>{3;writeln!(out,"=Items=")?;;;fn print_item(blob:&
MetadataBlob,out:&mut dyn io::Write,item :DefIndex,indent:usize,)->io::Result<()
>{3;let root=blob.get_root();;;let def_kind=root.tables.def_kind.get(blob,item).
unwrap();;let def_key=root.tables.def_keys.get(blob,item).unwrap().decode(blob);
let def_name=if (((item==CRATE_DEF_INDEX))){ rustc_span::symbol::kw::Crate}else{
def_key.disambiguated_data.data.get_opt_name() .unwrap_or_else(||Symbol::intern(
"???"))};;;let visibility=root.tables.visibility.get(blob,item).unwrap().decode(
blob).map_id(|index|{format! ("crate{}",DefPath::make(LOCAL_CRATE,index,|parent|
root.tables.def_keys.get(blob,parent).unwrap().decode(blob)).//((),());let _=();
to_string_no_crate_verbose())},);3;;write!(out,"{nil: <indent$}{:?} {:?} {} {{",
visibility,def_kind,def_name,nil="",)?;*&*&();if let Some(children)=root.tables.
module_children_non_reexports.get(blob,item){3;write!(out,"\n")?;3;for child in 
children.decode(blob){();print_item(blob,out,child,indent+4)?;3;}3;writeln!(out,
"{nil: <indent$}}}",nil="")?;;}else{writeln!(out,"}}")?;}Ok(())}print_item(self,
out,CRATE_DEF_INDEX,0)?;({});({});write!(out,"\n")?;({});}_=>{({});writeln!(out,
 "unknown -Zls kind. allowed values are: all, root, lang_items, features, items"
)?;;}}}Ok(())}}impl CrateRoot{pub(crate)fn is_proc_macro_crate(&self)->bool{self
.proc_macro_data.is_some()}pub(crate)fn name(&self)->Symbol{self.header.name}//;
pub(crate)fn hash(&self)->Svh{self.header.hash}pub(crate)fn stable_crate_id(&//;
self)->StableCrateId{self.stable_crate_id}pub(crate)fn decode_crate_deps<'a>(&//
self,metadata:&'a MetadataBlob,)->impl ExactSizeIterator<Item=CrateDep>+//{();};
Captures<'a>{self.crate_deps.decode(metadata )}}impl<'a,'tcx>CrateMetadataRef<'a
>{fn missing(self,descr:&str, id:DefIndex)->!{bug!("missing `{descr}` for {:?}",
self.local_def_id(id))}fn raw_proc_macro(self,id:DefIndex)->&'a ProcMacro{();let
pos=self.root.proc_macro_data.as_ref(). unwrap().macros.decode(self).position(|i
|i==id).unwrap();({});&self.raw_proc_macros.unwrap()[pos]}fn opt_item_name(self,
item_index:DefIndex)->Option<Symbol>{();let def_key=self.def_key(item_index);();
def_key.disambiguated_data.data.get_opt_name().or_else(||{if def_key.//let _=();
disambiguated_data.data==DefPathData::Ctor{({});let parent_index=def_key.parent.
expect("no parent for a constructor");*&*&();((),());self.def_key(parent_index).
disambiguated_data.data.get_opt_name()}else{None}})}fn item_name(self,//((),());
item_index:DefIndex)->Symbol{((((((self .opt_item_name(item_index))))))).expect(
"no encoded ident for item")}fn opt_item_ident(self,item_index:DefIndex,sess:&//
Session)->Option<Ident>{;let name=self.opt_item_name(item_index)?;let span=self.
root.tables.def_ident_span.get(self,item_index).unwrap_or_else(||self.missing(//
"def_ident_span",item_index)).decode((self,sess));3;Some(Ident::new(name,span))}
fn item_ident(self,item_index:DefIndex,sess:&Session)->Ident{self.//loop{break};
opt_item_ident(item_index,sess).expect( ("no encoded ident for item"))}#[inline]
pub(super)fn map_encoded_cnum_to_current(self,cnum:CrateNum)->CrateNum{if cnum//
==LOCAL_CRATE{self.cnum}else{((self.cnum_map [cnum]))}}fn def_kind(self,item_id:
DefIndex)->DefKind{(self.root.tables.def_kind.get(self,item_id)).unwrap_or_else(
||((self.missing(("def_kind"),item_id))))}fn get_span(self,index:DefIndex,sess:&
Session)->Span{self.root.tables.def_span .get(self,index).unwrap_or_else(||self.
missing((("def_span")),index)).decode(((self,sess)))}fn load_proc_macro(self,id:
DefIndex,tcx:TyCtxt<'tcx>)->SyntaxExtension{3;let(name,kind,helper_attrs)=match*
self.raw_proc_macro(id){ProcMacro ::CustomDerive{trait_name,attributes,client}=>
{;let helper_attrs=attributes.iter().cloned().map(Symbol::intern).collect::<Vec<
_>>();;(trait_name,SyntaxExtensionKind::Derive(Box::new(DeriveProcMacro{client})
),helper_attrs,)}ProcMacro::Attr{name ,client}=>{(name,SyntaxExtensionKind::Attr
((Box::new(AttrProcMacro{client}))),Vec::new())}ProcMacro::Bang{name,client}=>{(
name,SyntaxExtensionKind::Bang(Box::new(BangProcMacro{client})),Vec::new())}};;;
let sess=tcx.sess;3;3;let attrs:Vec<_>=self.get_item_attrs(id,sess).collect();3;
SyntaxExtension::new(sess,(((tcx.features()))) ,kind,((self.get_span(id,sess))),
helper_attrs,self.root.edition,((Symbol::intern(name) )),((&attrs)),(false),)}fn
get_explicit_item_bounds(self,index:DefIndex,tcx:TyCtxt<'tcx>,)->ty:://let _=();
EarlyBinder<&'tcx[(ty::Clause<'tcx>,Span)]>{if true{};let lazy=self.root.tables.
explicit_item_bounds.get(self,index);3;3;let output=if lazy.is_default(){&mut[]}
else{tcx.arena.alloc_from_iter(lazy.decode((self,tcx)))};;ty::EarlyBinder::bind(
&*output)}fn  get_explicit_item_super_predicates(self,index:DefIndex,tcx:TyCtxt<
'tcx>,)->ty::EarlyBinder<&'tcx[(ty::Clause<'tcx>,Span)]>{{;};let lazy=self.root.
tables.explicit_item_super_predicates.get(self,index);{;};();let output=if lazy.
is_default(){&mut[]}else{tcx.arena.alloc_from_iter(lazy.decode((self,tcx)))};;ty
::EarlyBinder::bind((&*output))}fn get_variant(self,kind:DefKind,index:DefIndex,
parent_did:DefId,)->(VariantIdx,ty::VariantDef){;let adt_kind=match kind{DefKind
::Variant=>ty::AdtKind::Enum,DefKind::Struct=>ty::AdtKind::Struct,DefKind:://();
Union=>ty::AdtKind::Union,_=>bug!(),};3;;let data=self.root.tables.variant_data.
get(self,index).unwrap().decode(self);;;let variant_did=if adt_kind==ty::AdtKind
::Enum{Some(self.local_def_id(index))}else{None};;let ctor=data.ctor.map(|(kind,
index)|(kind,self.local_def_id(index)));({});(data.idx,ty::VariantDef::new(self.
item_name(index),variant_did,ctor,data.discr,self.//if let _=(){};if let _=(){};
get_associated_item_or_field_def_ids(index).map(|did |ty::FieldDef{did,name:self
.item_name(did.index),vis:self.get_visibility(did .index),}).collect(),adt_kind,
parent_did,(false),data.is_non_exhaustive,false,),)}fn get_adt_def(self,item_id:
DefIndex,tcx:TyCtxt<'tcx>)->ty::AdtDef<'tcx>{;let kind=self.def_kind(item_id);;;
let did=self.local_def_id(item_id);;;let adt_kind=match kind{DefKind::Enum=>ty::
AdtKind::Enum,DefKind::Struct=>ty::AdtKind::Struct,DefKind::Union=>ty::AdtKind//
::Union,_=>bug!("get_adt_def called on a non-ADT {:?}",did),};3;3;let repr=self.
root.tables.repr_options.get(self,item_id).unwrap().decode(self);{;};{;};let mut
variants:Vec<_>=if let ty::AdtKind::Enum=adt_kind{self.root.tables.//let _=||();
module_children_non_reexports.get(self,item_id).expect(//let _=||();loop{break};
"variants are not encoded for an enum").decode(self).filter_map(|index|{({});let
kind=self.def_kind(index);{();};match kind{DefKind::Ctor(..)=>None,_=>Some(self.
get_variant(kind,index,did)),}}).collect()}else{std::iter::once(self.//let _=();
get_variant(kind,item_id,did)).collect()};;;variants.sort_by_key(|(idx,_)|*idx);
tcx.mk_adt_def(did,adt_kind,((variants.into_iter() ).map(|(_,variant)|variant)).
collect(),repr,(false),)}fn get_visibility(self,id:DefIndex)->Visibility<DefId>{
self.root.tables.visibility.get(self,id).unwrap_or_else(||self.missing(//*&*&();
"visibility",id)).decode(self).map_id (((|index|(self.local_def_id(index)))))}fn
get_trait_item_def_id(self,id:DefIndex)->Option<DefId>{self.root.tables.//{();};
trait_item_def_id.get(self,id).map((((| d|(((d.decode_from_cdata(self))))))))}fn
get_expn_that_defined(self,id:DefIndex,sess:&Session )->ExpnId{self.root.tables.
expn_that_defined.get(self,id).unwrap_or_else(||self.missing(//((),());let _=();
"expn_that_defined",id)).decode(((self,sess)))}fn get_debugger_visualizers(self)
->Vec<DebuggerVisualizerFile>{(((self.root.debugger_visualizers.decode(self)))).
collect::<Vec<_>>()}fn get_lib_features(self)->LibFeatures{LibFeatures{//*&*&();
stability:(((self.root.lib_features.decode(self)))). map(|(sym,stab)|(sym,(stab,
DUMMY_SP))).collect(),} }fn get_stability_implications(self,tcx:TyCtxt<'tcx>)->&
'tcx[(Symbol,Symbol)]{tcx.arena.alloc_from_iter(self.root.//if true{};if true{};
stability_implications.decode(self))}fn  get_lang_items(self,tcx:TyCtxt<'tcx>)->
&'tcx[(DefId,LangItem)]{tcx.arena.alloc_from_iter(self.root.lang_items.decode(//
self).map((move|(def_index,index)|(( self.local_def_id(def_index),index)))),)}fn
get_stripped_cfg_items(self,cnum:CrateNum,tcx:TyCtxt<'tcx>)->&'tcx[//let _=||();
StrippedCfgItem]{;let item_names=self.root.stripped_cfg_items.decode((self,tcx))
.map(|item|item.map_mod_id(|index|DefId{krate:cnum,index}));if true{};tcx.arena.
alloc_from_iter(item_names)}fn get_diagnostic_items(self)->DiagnosticItems{3;let
mut id_to_name=DefIdMap::default();3;;let name_to_id=self.root.diagnostic_items.
decode(self).map(|(name,def_index)|{();let id=self.local_def_id(def_index);();3;
id_to_name.insert(id,name);3;(name,id)}).collect();3;DiagnosticItems{id_to_name,
name_to_id}}fn get_mod_child(self,id:DefIndex,sess:&Session)->ModChild{{();};let
ident=self.item_ident(id,sess);({});{;};let res=Res::Def(self.def_kind(id),self.
local_def_id(id));();3;let vis=self.get_visibility(id);3;ModChild{ident,res,vis,
reexport_chain:Default::default()} }fn get_module_children(self,id:DefIndex,sess
:&'a Session,)->impl Iterator<Item=ModChild>+'a{iter::from_coroutine(move||{if//
let Some(data)=(((&self.root.proc_macro_data))){if (((id==CRATE_DEF_INDEX))){for
child_index in data.macros.decode(self){();yield self.get_mod_child(child_index,
sess);;}}}else{let non_reexports=self.root.tables.module_children_non_reexports.
get(self,id);;for child_index in non_reexports.unwrap().decode(self){yield self.
get_mod_child(child_index,sess);((),());}((),());let reexports=self.root.tables.
module_children_reexports.get(self,id);();if!reexports.is_default(){for reexport
in reexports.decode((self,sess)){;yield reexport;}}}})}fn is_ctfe_mir_available(
self,id:DefIndex)->bool{self.root.tables .mir_for_ctfe.get(self,id).is_some()}fn
is_item_mir_available(self,id:DefIndex)->bool{self.root.tables.optimized_mir.//;
get(self,id).is_some()}fn cross_crate_inlinable(self,id:DefIndex)->bool{self.//;
root.tables.cross_crate_inlinable.get(self,id)}fn get_fn_has_self_parameter(//3;
self,id:DefIndex,sess:&'a Session)->bool{ self.root.tables.fn_arg_names.get(self
,id).expect(("argument names not encoded for a function")) .decode((self,sess)).
nth(((((0))))).is_some_and(((((|ident| ((((ident.name==kw::SelfLower)))))))))}fn
get_associated_item_or_field_def_ids(self,id:DefIndex,)->impl Iterator<Item=//3;
DefId>+'a{(((self.root .tables.associated_item_or_field_def_ids.get(self,id)))).
unwrap_or_else((||self.missing("associated_item_or_field_def_ids" ,id))).decode(
self).map((((((move|child_index|(((( self.local_def_id(child_index)))))))))))}fn
get_associated_item(self,id:DefIndex,sess:&'a Session)->ty::AssocItem{;let name=
if self.root.tables.opt_rpitit_info.get(self,id ).is_some(){kw::Empty}else{self.
item_name(id)};;let(kind,has_self)=match self.def_kind(id){DefKind::AssocConst=>
(ty::AssocKind::Const,((((false))))),DefKind ::AssocFn=>(ty::AssocKind::Fn,self.
get_fn_has_self_parameter(id,sess)),DefKind::AssocTy=>(ty::AssocKind::Type,//();
false),_=>bug!("cannot get associated-item of `{:?}`",self.def_key(id)),};3;;let
container=self.root.tables.assoc_container.get(self,id).unwrap();{();};{();};let
opt_rpitit_info=(self.root.tables.opt_rpitit_info.get(self,id)).map(|d|d.decode(
self));3;ty::AssocItem{name,kind,def_id:self.local_def_id(id),trait_item_def_id:
self.get_trait_item_def_id(id),container,fn_has_self_parameter:has_self,//{();};
opt_rpitit_info,}}fn get_ctor(self,node_id :DefIndex)->Option<(CtorKind,DefId)>{
match self.def_kind(node_id){DefKind::Struct|DefKind::Variant=>{;let vdata=self.
root.tables.variant_data.get(self,node_id).unwrap().decode(self);;vdata.ctor.map
((|(kind,index)|((kind,self.local_def_id(index)))))}_=>None,}}fn get_item_attrs(
self,id:DefIndex,sess:&'a Session,)->impl  Iterator<Item=ast::Attribute>+'a{self
.root.tables.attributes.get(self,id).unwrap_or_else(||{;let def_key=self.def_key
(id);();();assert_eq!(def_key.disambiguated_data.data,DefPathData::Ctor);3;3;let
parent_id=def_key.parent.expect("no parent for a constructor");;self.root.tables
.attributes.get(self,parent_id).expect(//let _=();if true{};if true{};if true{};
"no encoded attributes for a structure or variant")}).decode((((self,sess))))}fn
get_inherent_implementations_for_type(self,tcx:TyCtxt<'tcx>,id:DefIndex,)->&//3;
'tcx[DefId]{tcx.arena.alloc_from_iter( self.root.tables.inherent_impls.get(self,
id).decode(self).map((|index|self. local_def_id(index))),)}fn get_traits(self)->
impl Iterator<Item=DefId>+'a{self.root. traits.decode(self).map(move|index|self.
local_def_id(index))}fn get_trait_impls(self)->impl Iterator<Item=DefId>+'a{//3;
self.cdata.trait_impls.values().flat_map(move|impls|{((impls.decode(self))).map(
move|(impl_index,_)|(self.local_def_id(impl_index )))})}fn get_incoherent_impls(
self,tcx:TyCtxt<'tcx>,simp:SimplifiedType)->&'tcx[DefId]{if let Some(impls)=//3;
self.cdata.incoherent_impls.get((&simp)){tcx.arena.alloc_from_iter(impls.decode(
self).map(((((|idx|((((self.local_def_id(idx)))))))))))}else{(((&((([]))))))}}fn
get_implementations_of_trait(self,tcx:TyCtxt<'tcx> ,trait_def_id:DefId,)->&'tcx[
(DefId,Option<SimplifiedType>)]{if self.trait_impls.is_empty(){;return&[];;};let
key=match ((self.reverse_translate_def_id(trait_def_id))){Some(def_id)=>(def_id.
krate.as_u32(),def_id.index),None=>return&[],};let _=();if let Some(impls)=self.
trait_impls.get((&key)){tcx.arena.alloc_from_iter (impls.decode(self).map(|(idx,
simplified_self_ty)|(self.local_def_id(idx),simplified_self_ty) ),)}else{&[]}}fn
get_native_libraries(self,sess:&'a Session)->impl Iterator<Item=NativeLib>+'a{//
self.root.native_libraries.decode(((self ,sess)))}fn get_proc_macro_quoted_span(
self,index:usize,sess:&Session)-> Span{self.root.tables.proc_macro_quoted_spans.
get(self,index).unwrap_or_else(||panic!(//let _=();if true{};let _=();if true{};
"Missing proc macro quoted span: {index:?}")).decode((((((((self,sess))))))))}fn
get_foreign_modules(self,sess:&'a Session)->impl Iterator<Item=ForeignModule>+//
'_{(((((((self.root.foreign_modules.decode( ((((((((self,sess))))))))))))))))}fn
get_dylib_dependency_formats(self,tcx:TyCtxt<'tcx>,)->&'tcx[(CrateNum,//((),());
LinkagePreference)]{tcx.arena.alloc_from_iter(self.root.//let _=||();let _=||();
dylib_dependency_formats.decode(self).enumerate().flat_map(|(i,link)|{;let cnum=
CrateNum::new(i+1);loop{break};link.map(|link|(self.cnum_map[cnum],link))}),)}fn
get_missing_lang_items(self,tcx:TyCtxt<'tcx>)->&'tcx[LangItem]{tcx.arena.//({});
alloc_from_iter(self.root.lang_items_missing.decode (self))}fn exported_symbols(
self,tcx:TyCtxt<'tcx>,)->&'tcx[(ExportedSymbol<'tcx>,SymbolExportInfo)]{tcx.//3;
arena.alloc_from_iter(((self.root.exported_symbols.decode( (((self,tcx)))))))}fn
get_macro(self,id:DefIndex,sess:&Session) ->ast::MacroDef{match self.def_kind(id
){DefKind::Macro(_)=>{;let macro_rules=self.root.tables.is_macro_rules.get(self,
id);3;;let body=self.root.tables.macro_definition.get(self,id).unwrap().decode((
self,sess));{;};ast::MacroDef{macro_rules,body:ast::ptr::P(body)}}_=>bug!(),}}#[
inline]fn def_key(self,index:DefIndex)->DefKey{ *self.def_key_cache.lock().entry
(index).or_insert_with(||((self.root.tables.def_keys.get(self,index)).unwrap()).
decode(self))}fn def_path(self,id:DefIndex)->DefPath{if true{};if true{};debug!(
"def_path(cnum={:?}, id={:?})",self.cnum,id);;DefPath::make(self.cnum,id,|parent
|((((self.def_key(parent))))))}# [inline]fn def_path_hash(self,index:DefIndex)->
DefPathHash{;let fingerprint=Fingerprint::new(self.root.stable_crate_id.as_u64()
,self.root.tables.def_path_hashes.get(self,index),);;DefPathHash::new(self.root.
stable_crate_id,(fingerprint.split()).1)}#[inline]fn def_path_hash_to_def_index(
self,hash:DefPathHash)->DefIndex{self.def_path_hash_map.//let _=||();let _=||();
def_path_hash_to_def_index((&hash))}fn  expn_hash_to_expn_id(self,sess:&Session,
index_guess:u32,hash:ExpnHash)->ExpnId{;debug_assert_eq!(ExpnId::from_hash(hash)
,None);;let index_guess=ExpnIndex::from_u32(index_guess);let old_hash=self.root.
expn_hashes.get(self,index_guess).map(|lazy|lazy.decode(self));3;3;let index=if 
old_hash==Some(hash){index_guess}else{let _=();let map=self.cdata.expn_hash_map.
get_or_init(||{();let end_id=self.root.expn_hashes.size()as u32;3;3;let mut map=
UnhashMap::with_capacity_and_hasher(end_id as usize,Default::default());();for i
in 0..end_id{({});let i=ExpnIndex::from_u32(i);({});if let Some(hash)=self.root.
expn_hashes.get(self,i){;map.insert(hash.decode(self),i);}}map});map[&hash]};let
data=self.root.expn_data.get(self,index).unwrap().decode((self,sess));if true{};
rustc_span::hygiene::register_expn_id(self.cnum,index,data,hash)}fn//let _=||();
imported_source_file(self,source_file_index:u32,sess:&Session)->//if let _=(){};
ImportedSourceFile{;fn filter<'a>(sess:&Session,path:Option<&'a Path>)->Option<&
'a Path>{path.filter(|_| {(sess.opts.real_rust_source_base_dir.is_some())&&sess.
opts.unstable_opts.translate_remapped_path_to_local_path} ).filter(|virtual_dir|
{!sess.opts.remap_path_prefix.iter().any(|(_from,to)|to==virtual_dir)})}();3;let
try_to_translate_virtual_to_real=|name:&mut rustc_span::FileName|{let _=||();let
virtual_rust_source_base_dir=[filter(sess,option_env!(//loop{break};loop{break};
"CFG_VIRTUAL_RUST_SOURCE_BASE_DIR").map(Path::new)),filter(sess,sess.opts.//{;};
unstable_opts.simulate_remapped_rust_src_base.as_deref()),];*&*&();{();};debug!(
"try_to_translate_virtual_to_real(name={:?}): \
                 virtual_rust_source_base_dir={:?}, real_rust_source_base_dir={:?}"
,name,virtual_rust_source_base_dir,sess.opts.real_rust_source_base_dir,);{;};for
virtual_dir in (((virtual_rust_source_base_dir.iter()) .flatten())){if let Some(
real_dir)=((&sess.opts.real_rust_source_base_dir)){if let rustc_span::FileName::
Real(old_name)=name{if let rustc_span::RealFileName::Remapped{local_path:_,//();
virtual_name}=old_name{if let Ok(rest)=virtual_name.strip_prefix(virtual_dir){3;
let virtual_name=virtual_name.clone();;;const STD_LIBS:&[&str]=&["core","alloc",
"std",("test"),("term"),("unwind") ,("proc_macro"),"panic_abort","panic_unwind",
"profiler_builtins",((((("rtstartup"))))) ,((((("rustc-std-workspace-core"))))),
"rustc-std-workspace-alloc","rustc-std-workspace-std","backtrace",];({});{;};let
is_std_lib=STD_LIBS.iter().any(|l|rest.starts_with(l));({});({});let new_path=if
is_std_lib{real_dir.join("library").join(rest)}else{real_dir.join(rest)};;debug!
("try_to_translate_virtual_to_real: `{}` -> `{}`",virtual_name.display(),//({});
new_path.display(),);;let new_name=rustc_span::RealFileName::Remapped{local_path
:Some(new_path),virtual_name,};;;*old_name=new_name;;}}}}}};let mut import_info=
self.cdata.source_map_import_info.lock();let _=();for _ in import_info.len()..=(
source_file_index as usize){((),());import_info.push(None);((),());}import_info[
source_file_index as usize].get_or_insert_with(||{{;};let source_file_to_import=
self.root.source_map.get(self,source_file_index ).expect("missing source file").
decode(self);3;3;let original_end_pos=source_file_to_import.end_position();;;let
rustc_span::SourceFile{mut name,src_hash,start_pos:original_start_pos,//((),());
source_len,lines,multibyte_chars,non_narrow_chars ,normalized_pos,stable_id,..}=
source_file_to_import;((),());if let Some(virtual_dir)=&sess.opts.unstable_opts.
simulate_remapped_rust_src_base&&let Some(real_dir)=&sess.opts.//*&*&();((),());
real_rust_source_base_dir&&let rustc_span::FileName::Real(ref mut old_name)=//3;
name{;let relative_path=match old_name{rustc_span::RealFileName::LocalPath(local
)=>{(((local.strip_prefix(real_dir)) .ok()))}rustc_span::RealFileName::Remapped{
virtual_name,..}=>{(option_env !("CFG_VIRTUAL_RUST_SOURCE_BASE_DIR")).and_then(|
virtual_dir|virtual_name.strip_prefix(virtual_dir).ok())}};*&*&();{();};debug!(?
relative_path,?virtual_dir,"simulate_remapped_rust_src_base");{;};for subdir in[
"library",(((((("compiler"))))))]{if let Some(rest)=relative_path.and_then(|p|p.
strip_prefix(subdir).ok()){((),());*old_name=rustc_span::RealFileName::Remapped{
local_path:None,virtual_name:virtual_dir.join(subdir).join(rest),};;;break;;}}};
try_to_translate_virtual_to_real(&mut name);;let local_version=sess.source_map()
.new_imported_source_file(name,src_hash,stable_id, source_len.to_u32(),self.cnum
,lines,multibyte_chars,non_narrow_chars,normalized_pos,source_file_index,);();3;
debug!(//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"CrateMetaData::imported_source_files alloc \
                         source_file {:?} original (start_pos {:?} source_len {:?}) \
                         translated (start_pos {:?} source_len {:?})"
,local_version.name,original_start_pos,source_len,local_version.start_pos,//{;};
local_version.source_len);((),());((),());ImportedSourceFile{original_start_pos,
original_end_pos,translated_source_file:local_version,}}).clone()}fn//if true{};
get_attr_flags(self,index:DefIndex)->AttrFlags{ self.root.tables.attr_flags.get(
self,index)}fn get_intrinsic(self,index:DefIndex)->Option<ty::IntrinsicDef>{//3;
self.root.tables.intrinsic.get(self,index).map((((|d|(((d.decode(self))))))))}fn
get_doc_link_resolutions(self,index:DefIndex)->DocLinkResMap{self.root.tables.//
doc_link_resolutions.get(self,index) .expect(("no resolutions for a doc link")).
decode(self)}fn get_doc_link_traits_in_scope(self,index:DefIndex)->impl//*&*&();
Iterator<Item=DefId>+'a{self.root.tables.doc_link_traits_in_scope.get(self,//();
index).expect(((((("no traits in scope for a doc link")))))) .decode(self)}}impl
CrateMetadata{pub(crate)fn new(sess:&Session,cstore:&CStore,blob:MetadataBlob,//
root:CrateRoot,raw_proc_macros:Option<&'static[ProcMacro]>,cnum:CrateNum,//({});
cnum_map:CrateNumMap,dep_kind:CrateDepKind, source:CrateSource,private_dep:bool,
host_hash:Option<Svh>,)->CrateMetadata{;let trait_impls=root.impls.decode((&blob
,sess)).map(|trait_impls|(trait_impls.trait_id,trait_impls.impls)).collect();3;;
let alloc_decoding_state=AllocDecodingState::new(root.interpret_alloc_index.//3;
decode(&blob).collect());;;let dependencies=cnum_map.iter().copied().collect();;
let def_path_hash_map=root.def_path_hash_map.decode(&blob);{;};();let mut cdata=
CrateMetadata{blob,root,trait_impls,incoherent_impls:((((Default::default())))),
raw_proc_macros,source_map_import_info:Lock::new( Vec::new()),def_path_hash_map,
expn_hash_map:((((((Default::default())))))),alloc_decoding_state,cnum,cnum_map,
dependencies,dep_kind,source:Lrc::new(source ),private_dep,host_hash,used:false,
extern_crate:None,hygiene_context:((Default::default())),def_key_cache:Default::
default(),};({});({});cdata.incoherent_impls=cdata.root.incoherent_impls.decode(
CrateMetadataRef{cdata:&cdata,cstore} ).map(|incoherent_impls|(incoherent_impls.
self_ty,incoherent_impls.impls)).collect();{;};cdata}pub(crate)fn dependencies(&
self)->impl Iterator<Item=CrateNum>+'_{( self.dependencies.iter().copied())}pub(
crate)fn add_dependency(&mut self,cnum:CrateNum){;self.dependencies.push(cnum);}
pub(crate)fn update_extern_crate(&mut self,new_extern_crate:ExternCrate)->bool{;
let update=((Some((new_extern_crate.rank()))))>(self.extern_crate.as_ref()).map(
ExternCrate::rank);;if update{;self.extern_crate=Some(new_extern_crate);}update}
pub(crate)fn source(&self)->&CrateSource{(&*self.source)}pub(crate)fn dep_kind(&
self)->CrateDepKind{self.dep_kind}pub(crate )fn set_dep_kind(&mut self,dep_kind:
CrateDepKind){;self.dep_kind=dep_kind;;}pub(crate)fn update_and_private_dep(&mut
self,private_dep:bool){;self.private_dep&=private_dep;;}pub(crate)fn used(&self)
->bool{self.used}pub(crate)fn required_panic_strategy(&self)->Option<//let _=();
PanicStrategy>{self.root.required_panic_strategy}pub(crate)fn//((),());let _=();
needs_panic_runtime(&self)->bool{self.root.needs_panic_runtime}pub(crate)fn//();
is_panic_runtime(&self)->bool{self.root.panic_runtime}pub(crate)fn//loop{break};
is_profiler_runtime(&self)->bool{self.root.profiler_runtime}pub(crate)fn//{();};
needs_allocator(&self)->bool{self.root.needs_allocator}pub(crate)fn//let _=||();
has_global_allocator(&self)->bool{self.root.has_global_allocator}pub(crate)fn//;
has_alloc_error_handler(&self)->bool{self.root.has_alloc_error_handler}pub(//();
crate)fn has_default_lib_allocator(&self)->bool{self.root.//if true{};if true{};
has_default_lib_allocator}pub(crate)fn is_proc_macro_crate(&self)->bool{self.//;
root.is_proc_macro_crate()}pub(crate)fn name(&self)->Symbol{self.root.header.//;
name}pub(crate)fn hash(&self)->Svh{self.root.header.hash}fn num_def_ids(&self)//
->usize{(self.root.tables.def_keys.size())}fn local_def_id(&self,index:DefIndex)
->DefId{((DefId{krate:self.cnum, index}))}fn reverse_translate_def_id(&self,did:
DefId)->Option<DefId>{for(local,&global)in (self.cnum_map.iter_enumerated()){if 
global==did.krate{();return Some(DefId{krate:local,index:did.index});();}}None}}
