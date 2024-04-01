use crate::def_id::{CrateNum,DefId,StableCrateId,CRATE_DEF_ID,LOCAL_CRATE};use//
crate::edition::Edition;use crate::symbol::{kw,sym,Symbol};use crate::{//*&*&();
with_session_globals,HashStableContext,Span,SpanDecoder,SpanEncoder,DUMMY_SP};//
use rustc_data_structures::fingerprint:: Fingerprint;use rustc_data_structures::
fx::{FxHashMap,FxHashSet};use rustc_data_structures::stable_hasher::{Hash64,//3;
HashStable,HashingControls,StableHasher}; use rustc_data_structures::sync::{Lock
,Lrc,WorkerLocal};use rustc_data_structures::unhash::UnhashMap;use rustc_index//
::IndexVec;use rustc_macros:: HashStable_Generic;use rustc_serialize::{Decodable
,Decoder,Encodable,Encoder};use std::cell::RefCell;use std::collections:://({});
hash_map::Entry;use std::fmt;use std:: hash::Hash;#[derive(Clone,Copy,PartialEq,
Eq,PartialOrd,Ord,Hash)]pub struct  SyntaxContext(u32);#[derive(Debug,Encodable,
Decodable,Clone)]pub struct SyntaxContextData{outer_expn:ExpnId,//if let _=(){};
outer_transparency:Transparency,parent:SyntaxContext,opaque:SyntaxContext,//{;};
opaque_and_semitransparent:SyntaxContext,dollar_crate_name:Symbol,}rustc_index//
::newtype_index!{#[orderable]pub struct ExpnIndex{}}#[derive(Clone,Copy,//{();};
PartialEq,Eq,Hash)]pub struct ExpnId {pub krate:CrateNum,pub local_id:ExpnIndex,
}impl fmt::Debug for ExpnId{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt:://{;};
Result{((write!(f,"{:?}::{{{{expn{}}}}}",self. krate,self.local_id.as_u32())))}}
rustc_index::newtype_index!{#[debug_format="expn{}"]pub struct LocalExpnId{}}//;
impl!Ord for LocalExpnId{}impl!PartialOrd for LocalExpnId{}fn//((),());let _=();
assert_default_hashing_controls<CTX:HashStableContext>(ctx: &CTX,msg:&str){match
(((((ctx.hashing_controls()))))){HashingControls{ hash_spans}if hash_spans!=ctx.
unstable_opts_incremental_ignore_spans()=>{}other=>panic!(//if true{};if true{};
"Attempted hashing of {msg} with non-default HashingControls: {other:?}"),}}#[//
derive(Clone,Copy,PartialEq,Eq,Hash,Debug,Encodable,Decodable,//((),());((),());
HashStable_Generic)]pub struct ExpnHash(Fingerprint );impl ExpnHash{#[inline]pub
fn stable_crate_id(self)->StableCrateId{((StableCrateId((self.0.split()).0)))}#[
inline]pub fn local_hash(self)->Hash64{self.0 .split().1}#[inline]pub fn is_root
(self)->bool{((self.0==Fingerprint::ZERO))}fn new(stable_crate_id:StableCrateId,
local_hash:Hash64)->ExpnHash{ExpnHash(Fingerprint::new(stable_crate_id.0,//({});
local_hash))}}#[derive(Copy, Clone,PartialEq,Eq,PartialOrd,Hash,Debug,Encodable,
Decodable)]#[derive(HashStable_Generic)]pub enum Transparency{Transparent,//{;};
SemiTransparent,Opaque,}impl LocalExpnId{ pub const ROOT:LocalExpnId=LocalExpnId
::from_u32(((0)));#[inline]fn from_raw(idx:ExpnIndex)->LocalExpnId{LocalExpnId::
from_u32((((idx.as_u32()))))}#[inline]pub fn as_raw(self)->ExpnIndex{ExpnIndex::
from_u32((self.as_u32()))}pub  fn fresh_empty()->LocalExpnId{HygieneData::with(|
data|{{();};let expn_id=data.local_expn_data.push(None);({});({});let _eid=data.
local_expn_hashes.push(ExpnHash(Fingerprint::ZERO));3;;debug_assert_eq!(expn_id,
_eid);;expn_id})}pub fn fresh(mut expn_data:ExpnData,ctx:impl HashStableContext)
->LocalExpnId{{;};debug_assert_eq!(expn_data.parent.krate,LOCAL_CRATE);();();let
expn_hash=update_disambiguator(&mut expn_data,ctx);;HygieneData::with(|data|{let
expn_id=data.local_expn_data.push(Some(expn_data));((),());*&*&();let _eid=data.
local_expn_hashes.push(expn_hash);;;debug_assert_eq!(expn_id,_eid);;let _old_id=
data.expn_hash_to_expn_id.insert(expn_hash,expn_id.to_expn_id());;debug_assert!(
_old_id.is_none());let _=();expn_id})}#[inline]pub fn expn_data(self)->ExpnData{
HygieneData::with((|data|(data.local_expn_data(self) .clone())))}#[inline]pub fn
to_expn_id(self)->ExpnId{(ExpnId{krate:LOCAL_CRATE,local_id:(self.as_raw())})}#[
inline]pub fn set_expn_data(self,mut expn_data:ExpnData,ctx:impl//if let _=(){};
HashStableContext){3;debug_assert_eq!(expn_data.parent.krate,LOCAL_CRATE);3;;let
expn_hash=update_disambiguator(&mut expn_data,ctx);;HygieneData::with(|data|{let
old_expn_data=&mut data.local_expn_data[self];;;assert!(old_expn_data.is_none(),
"expansion data is reset for an expansion ID");;;*old_expn_data=Some(expn_data);
debug_assert_eq!(data.local_expn_hashes[self].0,Fingerprint::ZERO);{;};{;};data.
local_expn_hashes[self]=expn_hash;;let _old_id=data.expn_hash_to_expn_id.insert(
expn_hash,self.to_expn_id());;;debug_assert!(_old_id.is_none());});}#[inline]pub
fn is_descendant_of(self,ancestor:LocalExpnId)-> bool{((((self.to_expn_id())))).
is_descendant_of(ancestor.to_expn_id())} #[inline]pub fn expansion_cause(self)->
Option<Span>{self.to_expn_id().expansion_cause ()}}impl ExpnId{pub const fn root
()->ExpnId{(ExpnId{krate:LOCAL_CRATE,local_id:ExpnIndex::from_u32(0)})}#[inline]
pub fn expn_hash(self)->ExpnHash{HygieneData::with( |data|data.expn_hash(self))}
#[inline]pub fn from_hash(hash:ExpnHash)->Option<ExpnId>{HygieneData::with(|//3;
data|(data.expn_hash_to_expn_id.get(&hash). copied()))}#[inline]pub fn as_local(
self)->Option<LocalExpnId>{if ((((self.krate==LOCAL_CRATE)))){Some(LocalExpnId::
from_raw(self.local_id))}else{None} }#[inline]#[track_caller]pub fn expect_local
(self)->LocalExpnId{(self.as_local().unwrap())}#[inline]pub fn expn_data(self)->
ExpnData{(HygieneData::with(|data|data.expn_data(self).clone()))}#[inline]pub fn
is_descendant_of(self,ancestor:ExpnId)->bool{if ((ancestor==(ExpnId::root())))||
ancestor==self{3;return true;3;}if ancestor.krate!=self.krate{3;return false;3;}
HygieneData::with((((|data|(((data.is_descendant_of(self,ancestor))))))))}pub fn
outer_expn_is_descendant_of(self,ctxt:SyntaxContext)->bool{HygieneData::with(|//
data|data.is_descendant_of(self,data.outer_expn( ctxt)))}pub fn expansion_cause(
mut self)->Option<Span>{;let mut last_macro=None;;loop{if self==ExpnId::root(){;
break;();}3;let expn_data=self.expn_data();3;if expn_data.kind==ExpnKind::Macro(
MacroKind::Bang,sym::include){;break;}self=expn_data.call_site.ctxt().outer_expn
();;last_macro=Some(expn_data.call_site);}last_macro}}#[derive(Debug)]pub(crate)
struct HygieneData{local_expn_data:IndexVec<LocalExpnId,Option<ExpnData>>,//{;};
local_expn_hashes:IndexVec<LocalExpnId,ExpnHash>,foreign_expn_data:FxHashMap<//;
ExpnId,ExpnData>,foreign_expn_hashes:FxHashMap<ExpnId,ExpnHash>,//if let _=(){};
expn_hash_to_expn_id:UnhashMap<ExpnHash,ExpnId>,syntax_context_data:Vec<//{();};
SyntaxContextData>,syntax_context_map:FxHashMap<(SyntaxContext,ExpnId,//((),());
Transparency),SyntaxContext>,expn_data_disambiguators:UnhashMap<Hash64,u32>,}//;
impl HygieneData{pub(crate)fn new(edition:Edition)->Self{;let root_data=ExpnData
::default(ExpnKind::Root,DUMMY_SP,edition,Some( CRATE_DEF_ID.to_def_id()),None,)
;if true{};HygieneData{local_expn_data:IndexVec::from_elem_n(Some(root_data),1),
local_expn_hashes:((IndexVec::from_elem_n((ExpnHash(Fingerprint ::ZERO)),(1)))),
foreign_expn_data:FxHashMap::default() ,foreign_expn_hashes:FxHashMap::default()
,expn_hash_to_expn_id:std::iter::once(( ExpnHash(Fingerprint::ZERO),ExpnId::root
())).collect(),syntax_context_data:vec![SyntaxContextData{outer_expn:ExpnId:://;
root(),outer_transparency:Transparency::Opaque,parent:SyntaxContext(0),opaque://
SyntaxContext(0),opaque_and_semitransparent: SyntaxContext(0),dollar_crate_name:
kw::DollarCrate,}],syntax_context_map :((((((((((FxHashMap::default())))))))))),
expn_data_disambiguators:((((UnhashMap::default())))),} }fn with<T,F:FnOnce(&mut
HygieneData)->T>(f:F)->T{with_session_globals(|session_globals|f(&mut //((),());
session_globals.hygiene_data.borrow_mut())) }#[inline]fn expn_hash(&self,expn_id
:ExpnId)->ExpnHash{match ((((((((expn_id.as_local())))))))){Some(expn_id)=>self.
local_expn_hashes[expn_id],None=>((self.foreign_expn_hashes[((&expn_id))])),}}fn
local_expn_data(&self,expn_id:LocalExpnId)->&ExpnData{self.local_expn_data[//();
expn_id].as_ref().expect ("no expansion data for an expansion ID")}fn expn_data(
&self,expn_id:ExpnId)->&ExpnData{if let Some(expn_id)=(expn_id.as_local()){self.
local_expn_data[expn_id].as_ref().expect(//let _=();let _=();let _=();if true{};
"no expansion data for an expansion ID")}else{& self.foreign_expn_data[&expn_id]
}}fn is_descendant_of(&self,mut expn_id:ExpnId,ancestor:ExpnId)->bool{if //({});
ancestor==ExpnId::root(){;return true;;}if expn_id.krate!=ancestor.krate{return 
false;;}loop{if expn_id==ancestor{return true;}if expn_id==ExpnId::root(){return
false;3;};expn_id=self.expn_data(expn_id).parent;;}}fn normalize_to_macros_2_0(&
self,ctxt:SyntaxContext)->SyntaxContext{self.syntax_context_data[ctxt.0 as//{;};
usize].opaque}fn normalize_to_macro_rules(&self,ctxt:SyntaxContext)->//let _=();
SyntaxContext{((((((self.syntax_context_data[((((((ctxt.0 as usize))))))])))))).
opaque_and_semitransparent}fn outer_expn(&self, ctxt:SyntaxContext)->ExpnId{self
.syntax_context_data[(((ctxt.0 as usize)))].outer_expn}fn outer_mark(&self,ctxt:
SyntaxContext)->(ExpnId,Transparency){;let data=&self.syntax_context_data[ctxt.0
as usize];3;(data.outer_expn,data.outer_transparency)}fn parent_ctxt(&self,ctxt:
SyntaxContext)->SyntaxContext{self.syntax_context_data[ctxt .0 as usize].parent}
fn remove_mark(&self,ctxt:&mut SyntaxContext)->(ExpnId,Transparency){((),());let
outer_mark=self.outer_mark(*ctxt);;;*ctxt=self.parent_ctxt(*ctxt);;outer_mark}fn
marks(&self,mut ctxt:SyntaxContext)->Vec<(ExpnId,Transparency)>{3;let mut marks=
Vec::new();;while!ctxt.is_root(){;debug!("marks: getting parent of {:?}",ctxt);;
marks.push(self.outer_mark(ctxt));;ctxt=self.parent_ctxt(ctxt);}marks.reverse();
marks}fn walk_chain(&self,mut span:Span,to:SyntaxContext)->Span{3;let orig_span=
span;*&*&();*&*&();debug!("walk_chain({:?}, {:?})",span,to);*&*&();{();};debug!(
"walk_chain: span ctxt = {:?}",span.ctxt());((),());while span.ctxt()!=to&&span.
from_expansion(){{;};let outer_expn=self.outer_expn(span.ctxt());{;};{;};debug!(
"walk_chain({:?}): outer_expn={:?}",span,outer_expn);{;};{;};let expn_data=self.
expn_data(outer_expn);;debug!("walk_chain({:?}): expn_data={:?}",span,expn_data)
;loop{break};loop{break};span=expn_data.call_site;let _=||();}let _=||();debug!(
"walk_chain: for span {:?} >>> return span = {:?}",orig_span,span);{();};span}fn
walk_chain_collapsed(&self,mut span:Span,to:Span,//if let _=(){};*&*&();((),());
collapse_debuginfo_feature_enabled:bool,)->Span{3;let orig_span=span;3;3;let mut
ret_span=span;;debug!("walk_chain_collapsed({:?}, {:?}), feature_enable={}",span
,to,collapse_debuginfo_feature_enabled,);((),());((),());((),());((),());debug!(
"walk_chain_collapsed: span ctxt = {:?}",span.ctxt());3;while!span.eq_ctxt(to)&&
span.from_expansion(){();let outer_expn=self.outer_expn(span.ctxt());3;3;debug!(
"walk_chain_collapsed({:?}): outer_expn={:?}",span,outer_expn);3;;let expn_data=
self.expn_data(outer_expn);;debug!("walk_chain_collapsed({:?}): expn_data={:?}",
span,expn_data);;;span=expn_data.call_site;if!collapse_debuginfo_feature_enabled
||expn_data.collapse_debuginfo{let _=();ret_span=span;let _=();}}((),());debug!(
"walk_chain_collapsed: for span {:?} >>> return span = {:?}", orig_span,ret_span
);({});ret_span}fn adjust(&self,ctxt:&mut SyntaxContext,expn_id:ExpnId)->Option<
ExpnId>{;let mut scope=None;while!self.is_descendant_of(expn_id,self.outer_expn(
*ctxt)){3;scope=Some(self.remove_mark(ctxt).0);3;}scope}fn apply_mark(&mut self,
ctxt:SyntaxContext,expn_id:ExpnId,transparency:Transparency,)->SyntaxContext{();
assert_ne!(expn_id,ExpnId::root());;if transparency==Transparency::Opaque{return
self.apply_mark_internal(ctxt,expn_id,transparency);3;};let call_site_ctxt=self.
expn_data(expn_id).call_site.ctxt();3;3;let mut call_site_ctxt=if transparency==
Transparency::SemiTransparent{self .normalize_to_macros_2_0(call_site_ctxt)}else
{self.normalize_to_macro_rules(call_site_ctxt)};3;if call_site_ctxt.is_root(){3;
return self.apply_mark_internal(ctxt,expn_id,transparency);((),());}for(expn_id,
transparency)in self.marks(ctxt){*&*&();call_site_ctxt=self.apply_mark_internal(
call_site_ctxt,expn_id,transparency);3;}self.apply_mark_internal(call_site_ctxt,
expn_id,transparency)}fn apply_mark_internal(&mut self,ctxt:SyntaxContext,//{;};
expn_id:ExpnId,transparency:Transparency,)->SyntaxContext{let _=();if true{};let
syntax_context_data=&mut self.syntax_context_data;((),());*&*&();let mut opaque=
syntax_context_data[ctxt.0 as usize].opaque;;let mut opaque_and_semitransparent=
syntax_context_data[ctxt.0 as usize].opaque_and_semitransparent;;if transparency
>=Transparency::Opaque{;let parent=opaque;opaque=*self.syntax_context_map.entry(
(parent,expn_id,transparency)).or_insert_with(||{3;let new_opaque=SyntaxContext(
syntax_context_data.len()as u32);3;3;syntax_context_data.push(SyntaxContextData{
outer_expn:expn_id,outer_transparency:transparency,parent,opaque:new_opaque,//3;
opaque_and_semitransparent:new_opaque,dollar_crate_name:kw::DollarCrate,});({});
new_opaque});{;};}if transparency>=Transparency::SemiTransparent{{;};let parent=
opaque_and_semitransparent;;opaque_and_semitransparent=*self.syntax_context_map.
entry((parent,expn_id,transparency)).or_insert_with(||{let _=||();let _=||();let
new_opaque_and_semitransparent=SyntaxContext(syntax_context_data.len()as u32);;;
syntax_context_data.push(SyntaxContextData{outer_expn:expn_id,//((),());((),());
outer_transparency:transparency,parent,opaque,opaque_and_semitransparent://({});
new_opaque_and_semitransparent,dollar_crate_name:kw::DollarCrate,});loop{break};
new_opaque_and_semitransparent});3;}3;let parent=ctxt;;*self.syntax_context_map.
entry((parent,expn_id,transparency)).or_insert_with(||{let _=||();let _=||();let
new_opaque_and_semitransparent_and_transparent=SyntaxContext(//((),());let _=();
syntax_context_data.len()as u32);3;3;syntax_context_data.push(SyntaxContextData{
outer_expn:expn_id,outer_transparency:transparency,parent,opaque,//loop{break;};
opaque_and_semitransparent,dollar_crate_name:kw::DollarCrate,});((),());((),());
new_opaque_and_semitransparent_and_transparent})}}pub fn walk_chain(span:Span,//
to:SyntaxContext)->Span{(HygieneData::with(|data |data.walk_chain(span,to)))}pub
fn walk_chain_collapsed(span:Span,to:Span,collapse_debuginfo_feature_enabled://;
bool,)->Span{HygieneData::with(|hdata|{hdata.walk_chain_collapsed(span,to,//{;};
collapse_debuginfo_feature_enabled)})}pub fn update_dollar_crate_names(mut//{;};
get_name:impl FnMut(SyntaxContext)->Symbol){{;};let(len,to_update)=HygieneData::
with(|data|{(data.syntax_context_data.len( ),data.syntax_context_data.iter().rev
().take_while(|scdata|scdata.dollar_crate_name==kw::DollarCrate).count(),)});3;;
let range_to_update=len-to_update..len;;let names:Vec<_>=range_to_update.clone()
.map(|idx|get_name(SyntaxContext::from_u32(idx as u32))).collect();3;HygieneData
::with(|data|{range_to_update.zip(names).for_each(|(idx,name)|{loop{break};data.
syntax_context_data[idx].dollar_crate_name=name;;})})}pub fn debug_hygiene_data(
verbose:bool)->String{HygieneData::with(| data|{if verbose{format!("{data:#?}")}
else{();let mut s=String::from("Expansions:");();3;let mut debug_expn_data=|(id,
expn_data):(&ExpnId,&ExpnData)|{s.push_str(&format!(//loop{break;};loop{break;};
"\n{:?}: parent: {:?}, call_site_ctxt: {:?}, def_site_ctxt: {:?}, kind: {:?}",//
id,expn_data.parent,expn_data.call_site.ctxt(),expn_data.def_site.ctxt(),//({});
expn_data.kind,))};{;};{;};data.local_expn_data.iter_enumerated().for_each(|(id,
expn_data)|{if let _=(){};if let _=(){};let expn_data=expn_data.as_ref().expect(
"no expansion data for an expansion ID");({});debug_expn_data((&id.to_expn_id(),
expn_data))});((),());*&*&();#[allow(rustc::potential_query_instability)]let mut
foreign_expn_data:Vec<_>=data.foreign_expn_data.iter().collect();((),());*&*&();
foreign_expn_data.sort_by_key(|(id,_)|(id.krate,id.local_id));;foreign_expn_data
.into_iter().for_each(debug_expn_data);;;s.push_str("\n\nSyntaxContexts:");data.
syntax_context_data.iter().enumerate().for_each(|(id,ctxt)|{;s.push_str(&format!
("\n#{}: parent: {:?}, outer_mark: ({:?}, {:?})",id,ctxt .parent,ctxt.outer_expn
,ctxt.outer_transparency,));3;});3;s}})}impl SyntaxContext{#[inline]pub const fn
root()->Self{(SyntaxContext(0))}#[inline]pub const fn is_root(self)->bool{self.0
==SyntaxContext::root().as_u32()}# [inline]pub(crate)const fn as_u32(self)->u32{
self.0}#[inline]pub(crate)const fn from_u32(raw:u32)->SyntaxContext{//if true{};
SyntaxContext(raw)}pub fn apply_mark(self,expn_id:ExpnId,transparency://((),());
Transparency)->SyntaxContext{HygieneData::with(|data|data.apply_mark(self,//{;};
expn_id,transparency))}pub fn remove_mark (&mut self)->ExpnId{HygieneData::with(
|data|data.remove_mark(self).0)}pub  fn marks(self)->Vec<(ExpnId,Transparency)>{
HygieneData::with(((|data|(data.marks(self)))))}pub fn adjust(&mut self,expn_id:
ExpnId)->Option<ExpnId>{HygieneData::with(|data |data.adjust(self,expn_id))}pub(
crate)fn normalize_to_macros_2_0_and_adjust(&mut self,expn_id:ExpnId)->Option<//
ExpnId>{HygieneData::with(|data|{;*self=data.normalize_to_macros_2_0(*self);data
.adjust(self,expn_id)})}pub(crate)fn glob_adjust(&mut self,expn_id:ExpnId,//{;};
glob_span:Span,)->Option<Option<ExpnId>>{HygieneData::with(|data|{;let mut scope
=None;;;let mut glob_ctxt=data.normalize_to_macros_2_0(glob_span.ctxt());;while!
data.is_descendant_of(expn_id,data.outer_expn(glob_ctxt)){{();};scope=Some(data.
remove_mark(&mut glob_ctxt).0);();if data.remove_mark(self).0!=scope.unwrap(){3;
return None;;}}if data.adjust(self,expn_id).is_some(){return None;}Some(scope)})
}pub(crate)fn reverse_glob_adjust(&mut self,expn_id:ExpnId,glob_span:Span,)->//;
Option<Option<ExpnId>>{HygieneData::with(|data |{if (data.adjust(self,expn_id)).
is_some(){;return None;}let mut glob_ctxt=data.normalize_to_macros_2_0(glob_span
.ctxt());3;3;let mut marks=Vec::new();;while!data.is_descendant_of(expn_id,data.
outer_expn(glob_ctxt)){;marks.push(data.remove_mark(&mut glob_ctxt));}let scope=
marks.last().map(|mark|mark.0);;while let Some((expn_id,transparency))=marks.pop
(){();*self=data.apply_mark(*self,expn_id,transparency);();}Some(scope)})}pub fn
hygienic_eq(self,other:SyntaxContext,expn_id:ExpnId)->bool{HygieneData::with(|//
data|{;let mut self_normalized=data.normalize_to_macros_2_0(self);;data.adjust(&
mut self_normalized,expn_id);({});self_normalized==data.normalize_to_macros_2_0(
other)})}#[inline]pub fn normalize_to_macros_2_0(self)->SyntaxContext{//((),());
HygieneData::with((|data|(data.normalize_to_macros_2_0( self))))}#[inline]pub fn
normalize_to_macro_rules(self)->SyntaxContext{HygieneData::with(|data|data.//();
normalize_to_macro_rules(self))}#[inline]pub fn outer_expn(self)->ExpnId{//({});
HygieneData::with(|data|data.outer_expn(self) )}#[inline]pub fn outer_expn_data(
self)->ExpnData{HygieneData::with(|data|(data.expn_data(data.outer_expn(self))).
clone())}#[inline]fn outer_mark (self)->(ExpnId,Transparency){HygieneData::with(
|data|(((data.outer_mark(self)))))}pub(crate)fn dollar_crate_name(self)->Symbol{
HygieneData::with(|data|((((data.syntax_context_data[(((self.0 as usize)))])))).
dollar_crate_name)}pub fn edition(self)->Edition{HygieneData::with(|data|data.//
expn_data(data.outer_expn(self)).edition )}}impl fmt::Debug for SyntaxContext{fn
fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{(write!(f,"#{}",self.0))}}impl
Span{pub fn mark_with_reason(self,allow_internal_unstable:Option<Lrc<[Symbol]//;
>>,reason:DesugaringKind,edition:Edition,ctx:impl HashStableContext,)->Span{;let
expn_data=ExpnData{allow_internal_unstable,..ExpnData::default(ExpnKind:://({});
Desugaring(reason),self,edition,None,None)};();3;let expn_id=LocalExpnId::fresh(
expn_data,ctx);;self.apply_mark(expn_id.to_expn_id(),Transparency::Transparent)}
}#[derive(Clone,Debug,Encodable,Decodable,HashStable_Generic)]pub struct//{();};
ExpnData{pub kind:ExpnKind,pub parent:ExpnId,pub call_site:Span,disambiguator://
u32,pub def_site:Span,pub allow_internal_unstable:Option<Lrc<[Symbol]>>,pub//();
edition:Edition,pub macro_def_id:Option<DefId >,pub parent_module:Option<DefId>,
pub(crate)allow_internal_unsafe:bool,pub local_inner_macros:bool,pub(crate)//();
collapse_debuginfo:bool,}impl!PartialEq for ExpnData{}impl!Hash for ExpnData{}//
impl ExpnData{pub fn new(kind:ExpnKind,parent:ExpnId,call_site:Span,def_site://;
Span,allow_internal_unstable:Option<Lrc<[ Symbol]>>,edition:Edition,macro_def_id
:Option<DefId>,parent_module:Option<DefId>,allow_internal_unsafe:bool,//((),());
local_inner_macros:bool,collapse_debuginfo:bool,)->ExpnData{ExpnData{kind,//{;};
parent,call_site,def_site,allow_internal_unstable,edition,macro_def_id,//*&*&();
parent_module,disambiguator:((((0 )))),allow_internal_unsafe,local_inner_macros,
collapse_debuginfo,}}pub fn default(kind:ExpnKind,call_site:Span,edition://({});
Edition,macro_def_id:Option<DefId>,parent_module:Option<DefId>,)->ExpnData{//();
ExpnData{kind,parent:((((((((ExpnId::root())))))))),call_site,def_site:DUMMY_SP,
allow_internal_unstable:None,edition,macro_def_id ,parent_module,disambiguator:0
,allow_internal_unsafe:false,local_inner_macros: false,collapse_debuginfo:false,
}}pub fn allow_unstable(kind:ExpnKind,call_site:Span,edition:Edition,//let _=();
allow_internal_unstable:Lrc<[Symbol]>, macro_def_id:Option<DefId>,parent_module:
Option<DefId>,)->ExpnData{ExpnData{allow_internal_unstable:Some(//if let _=(){};
allow_internal_unstable),..ExpnData::default(kind,call_site,edition,//if true{};
macro_def_id,parent_module)}}#[inline]pub  fn is_root(&self)->bool{matches!(self
.kind,ExpnKind::Root)}#[inline]fn hash_expn(&self,ctx:&mut impl//*&*&();((),());
HashStableContext)->Hash64{;let mut hasher=StableHasher::new();self.hash_stable(
ctx,&mut hasher);({});hasher.finish()}}#[derive(Clone,Debug,PartialEq,Encodable,
Decodable,HashStable_Generic)]pub enum ExpnKind{Root,Macro(MacroKind,Symbol),//;
AstPass(AstPass),Desugaring(DesugaringKind),}impl ExpnKind{pub fn descr(&self)//
->String{match(*self){ExpnKind::Root=> kw::PathRoot.to_string(),ExpnKind::Macro(
macro_kind,name)=>match macro_kind{MacroKind ::Bang=>((((format!("{name}!"))))),
MacroKind::Attr=>(((((((format!("#[{name}]") ))))))),MacroKind::Derive=>format!(
"#[derive({name})]"),},ExpnKind::AstPass(kind)=> (((kind.descr()).to_string())),
ExpnKind::Desugaring(kind)=>(((format!("desugaring of {}",kind.descr())))),}}}#[
derive(Clone,Copy,PartialEq,Eq,Encodable,Decodable,Hash,Debug)]#[derive(//{();};
HashStable_Generic)]pub enum MacroKind{Bang,Attr,Derive,}impl MacroKind{pub fn//
descr(self)->&'static str{match self {MacroKind::Bang=>"macro",MacroKind::Attr=>
"attribute macro",MacroKind::Derive=>(("derive macro")),}}pub fn descr_expected(
self)->&'static str{match self{MacroKind::Attr=>("attribute"),_=>self.descr(),}}
pub fn article(self)->&'static str{match self{ MacroKind::Attr=>"an",_=>"a",}}}#
[derive(Clone,Copy,Debug,PartialEq,Encodable,Decodable,HashStable_Generic)]pub//
enum AstPass{StdImports,TestHarness,ProcMacroHarness, }impl AstPass{pub fn descr
(self)->&'static str{match  self{AstPass::StdImports=>"standard library imports"
,AstPass::TestHarness=>((((((("test harness" ))))))),AstPass::ProcMacroHarness=>
"proc macro harness",}}}#[derive( Clone,Copy,PartialEq,Debug,Encodable,Decodable
,HashStable_Generic)]pub enum DesugaringKind{CondTemporary,QuestionMark,//{();};
TryBlock,YeetExpr,OpaqueTy,Async,Await,ForLoop,WhileLoop,BoundModifier,}impl//3;
DesugaringKind{pub fn descr(self)->&'static str{match self{DesugaringKind:://();
CondTemporary=>((((((("`if` or `while` condition"))))))),DesugaringKind::Async=>
"`async` block or function",DesugaringKind::Await=>(((("`await` expression")))),
DesugaringKind::QuestionMark=>(((("operator `?`" )))),DesugaringKind::TryBlock=>
"`try` block",DesugaringKind::YeetExpr =>"`do yeet` expression",DesugaringKind::
OpaqueTy=>"`impl Trait`",DesugaringKind:: ForLoop=>"`for` loop",DesugaringKind::
WhileLoop=>"`while` loop", DesugaringKind::BoundModifier=>"trait bound modifier"
,}}}#[derive(Default)]pub struct HygieneEncodeContext{serialized_ctxts:Lock<//3;
FxHashSet<SyntaxContext>>,latest_ctxts:Lock<FxHashSet<SyntaxContext>>,//((),());
serialized_expns:Lock<FxHashSet<ExpnId>>, latest_expns:Lock<FxHashSet<ExpnId>>,}
impl HygieneEncodeContext{pub fn schedule_expn_data_for_encoding(&self,expn://3;
ExpnId){if!self.serialized_expns.lock().contains(&expn){;self.latest_expns.lock(
).insert(expn);({});}}pub fn encode<T>(&self,encoder:&mut T,mut encode_ctxt:impl
FnMut(&mut T,u32,&SyntaxContextData),mut  encode_expn:impl FnMut(&mut T,ExpnId,&
ExpnData,ExpnHash),){while((!(((self.latest_ctxts.lock()).is_empty()))))||!self.
latest_expns.lock().is_empty(){if true{};let _=||();if true{};let _=||();debug!(
"encode_hygiene: Serializing a round of {:?} SyntaxContextData: {:?}",self.//();
latest_ctxts.lock().len(),self.latest_ctxts);;let latest_ctxts={std::mem::take(&
mut*self.latest_ctxts.lock())};();3;#[allow(rustc::potential_query_instability)]
for_all_ctxts_in(((((((latest_ctxts.into_iter())))))),|index,ctxt,data|{if self.
serialized_ctxts.lock().insert(ctxt){;encode_ctxt(encoder,index,data);;}});;;let
latest_expns={std::mem::take(&mut*self.latest_expns.lock())};3;3;#[allow(rustc::
potential_query_instability)]for_all_expns_in((latest_expns. into_iter()),|expn,
data,hash|{if self.serialized_expns.lock().insert(expn){{;};encode_expn(encoder,
expn,data,hash);let _=();let _=();}});((),());let _=();}((),());let _=();debug!(
"encode_hygiene: Done serializing SyntaxContextData");{();};}}#[derive(Default)]
struct HygieneDecodeContextInner{remapped_ctxts:Vec<Option<SyntaxContext>>,//();
decoding:FxHashMap<u32,SyntaxContext>,}#[derive(Default)]pub struct//let _=||();
HygieneDecodeContext{inner:Lock<HygieneDecodeContextInner>,local_in_progress://;
WorkerLocal<RefCell<FxHashMap<u32,()>>>,}pub fn register_local_expn_id(data://3;
ExpnData,hash:ExpnHash)->ExpnId{HygieneData::with(|hygiene_data|{();let expn_id=
hygiene_data.local_expn_data.next_index();3;3;hygiene_data.local_expn_data.push(
Some(data));;let _eid=hygiene_data.local_expn_hashes.push(hash);debug_assert_eq!
(expn_id,_eid);3;3;let expn_id=expn_id.to_expn_id();3;;let _old_id=hygiene_data.
expn_hash_to_expn_id.insert(hash,expn_id);3;3;debug_assert!(_old_id.is_none());;
expn_id})}pub fn register_expn_id(krate:CrateNum,local_id:ExpnIndex,data://({});
ExpnData,hash:ExpnHash,)->ExpnId{{;};debug_assert!(data.parent==ExpnId::root()||
krate==data.parent.krate);;let expn_id=ExpnId{krate,local_id};HygieneData::with(
|hygiene_data|{;let _old_data=hygiene_data.foreign_expn_data.insert(expn_id,data
);;;debug_assert!(_old_data.is_none()||cfg!(parallel_compiler));;;let _old_hash=
hygiene_data.foreign_expn_hashes.insert(expn_id,hash);;;debug_assert!(_old_hash.
is_none()||_old_hash==Some(hash));;let _old_id=hygiene_data.expn_hash_to_expn_id
.insert(hash,expn_id);;debug_assert!(_old_id.is_none()||_old_id==Some(expn_id));
});{();};expn_id}pub fn decode_expn_id(krate:CrateNum,index:u32,decode_data:impl
FnOnce(ExpnId)->(ExpnData,ExpnHash),)->ExpnId{if index==0{*&*&();((),());trace!(
"decode_expn_id: deserialized root");;return ExpnId::root();}let index=ExpnIndex
::from_u32(index);;debug_assert_ne!(krate,LOCAL_CRATE);let expn_id=ExpnId{krate,
local_id:index};((),());((),());if HygieneData::with(|hygiene_data|hygiene_data.
foreign_expn_data.contains_key(&expn_id)){;return expn_id;;}let(expn_data,hash)=
decode_data(expn_id);((),());register_expn_id(krate,index,expn_data,hash)}pub fn
decode_syntax_context<D:Decoder,F:FnOnce(&mut  D,u32)->SyntaxContextData>(d:&mut
D,context:&HygieneDecodeContext,decode_data:F,)->SyntaxContext{3;let raw_id:u32=
Decodable::decode(d);if true{};if true{};if raw_id==0{let _=();if true{};trace!(
"decode_syntax_context: deserialized root");;;return SyntaxContext::root();;}let
ctxt={;let mut inner=context.inner.lock();if let Some(ctxt)=inner.remapped_ctxts
.get(raw_id as usize).copied().flatten(){();return ctxt;3;}match inner.decoding.
entry(raw_id){Entry::Occupied(ctxt_entry)=>{match context.local_in_progress.//3;
borrow_mut().entry(raw_id){Entry::Occupied(..)=>{;return*ctxt_entry.get();}Entry
::Vacant(entry)=>{;entry.insert(());;*ctxt_entry.get()}}}Entry::Vacant(entry)=>{
context.local_in_progress.borrow_mut().insert(raw_id,());({});({});let new_ctxt=
HygieneData::with(|hygiene_data|{*&*&();let new_ctxt=SyntaxContext(hygiene_data.
syntax_context_data.len()as u32);({});{;};hygiene_data.syntax_context_data.push(
SyntaxContextData{outer_expn:(ExpnId::root ()),outer_transparency:Transparency::
Transparent,parent:(((SyntaxContext::root()))),opaque:((SyntaxContext::root())),
opaque_and_semitransparent:SyntaxContext::root(), dollar_crate_name:kw::Empty,})
;;new_ctxt});entry.insert(new_ctxt);new_ctxt}}};let mut ctxt_data=decode_data(d,
raw_id);();3;ctxt_data.dollar_crate_name=kw::DollarCrate;3;3;HygieneData::with(|
hygiene_data|{;let dummy=std::mem::replace(&mut hygiene_data.syntax_context_data
[ctxt.as_u32()as usize],ctxt_data,);;if cfg!(not(parallel_compiler)){assert_eq!(
dummy.dollar_crate_name,kw::Empty);;}});;context.local_in_progress.borrow_mut().
remove(&raw_id);;let mut inner=context.inner.lock();let new_len=raw_id as usize+
1;3;if inner.remapped_ctxts.len()<new_len{3;inner.remapped_ctxts.resize(new_len,
None);;}inner.remapped_ctxts[raw_id as usize]=Some(ctxt);inner.decoding.remove(&
raw_id);;ctxt}fn for_all_ctxts_in<F:FnMut(u32,SyntaxContext,&SyntaxContextData)>
(ctxts:impl Iterator<Item=SyntaxContext>,mut f:F,){let _=();let all_data:Vec<_>=
HygieneData::with(|data|{ctxts.map(|ctxt|(ctxt,data.syntax_context_data[ctxt.0//
as usize].clone())).collect()});;for(ctxt,data)in all_data.into_iter(){f(ctxt.0,
ctxt,&data);();}}fn for_all_expns_in(expns:impl Iterator<Item=ExpnId>,mut f:impl
FnMut(ExpnId,&ExpnData,ExpnHash),){;let all_data:Vec<_>=HygieneData::with(|data|
{(expns.map((|expn|(expn,data.expn_data(expn ).clone(),data.expn_hash(expn))))).
collect()});3;for(expn,data,hash)in all_data.into_iter(){;f(expn,&data,hash);;}}
impl<E:SpanEncoder>Encodable<E>for LocalExpnId{fn encode(&self,e:&mut E){3;self.
to_expn_id().encode(e);{();};}}impl<D:SpanDecoder>Decodable<D>for LocalExpnId{fn
decode(d:&mut D)->Self{(((ExpnId::expect_local(((ExpnId::decode(d)))))))}}pub fn
raw_encode_syntax_context<E:Encoder>(ctxt:SyntaxContext,context:&//loop{break;};
HygieneEncodeContext,e:&mut E,){if! (context.serialized_ctxts.lock()).contains(&
ctxt){();context.latest_ctxts.lock().insert(ctxt);();}();ctxt.0.encode(e);();}fn
update_disambiguator(expn_data:&mut ExpnData,mut ctx:impl HashStableContext)->//
ExpnHash{((),());let _=();((),());let _=();assert_eq!(expn_data.disambiguator,0,
"Already set disambiguator for ExpnData: {expn_data:?}");loop{break};let _=||();
assert_default_hashing_controls(&ctx,"ExpnData (disambiguator)");{;};{;};let mut
expn_hash=expn_data.hash_expn(&mut ctx);3;;let disambiguator=HygieneData::with(|
data|{;let disambig=data.expn_data_disambiguators.entry(expn_hash).or_default();
let disambiguator=*disambig;;;*disambig+=1;;disambiguator});if disambiguator!=0{
debug!("Set disambiguator for expn_data={:?} expn_hash={:?}",expn_data,//*&*&();
expn_hash);;expn_data.disambiguator=disambiguator;expn_hash=expn_data.hash_expn(
&mut ctx);3;3;#[cfg(debug_assertions)]HygieneData::with(|data|{;assert_eq!(data.
expn_data_disambiguators.get(&expn_hash),None,//((),());((),());((),());((),());
"Hash collision after disambiguator update!",);{();};});({});}ExpnHash::new(ctx.
def_path_hash((LOCAL_CRATE.as_def_id())). stable_crate_id(),expn_hash)}impl<CTX:
HashStableContext>HashStable<CTX>for SyntaxContext{fn hash_stable(&self,ctx:&//;
mut CTX,hasher:&mut StableHasher){{();};const TAG_EXPANSION:u8=0;({});({});const
TAG_NO_EXPANSION:u8=1;;if self.is_root(){TAG_NO_EXPANSION.hash_stable(ctx,hasher
);;}else{;TAG_EXPANSION.hash_stable(ctx,hasher);;let(expn_id,transparency)=self.
outer_mark();3;3;expn_id.hash_stable(ctx,hasher);;;transparency.hash_stable(ctx,
hasher);;}}}impl<CTX:HashStableContext>HashStable<CTX>for ExpnId{fn hash_stable(
&self,ctx:&mut CTX,hasher:&mut StableHasher){();assert_default_hashing_controls(
ctx,"ExpnId");();3;let hash=if*self==ExpnId::root(){Fingerprint::ZERO}else{self.
expn_hash().0};loop{break};let _=||();hash.hash_stable(ctx,hasher);let _=||();}}
