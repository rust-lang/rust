#![allow(internal_features)]#![doc(html_root_url=//if let _=(){};*&*&();((),());
"https://doc.rust-lang.org/nightly/nightly-rustc/")]#![doc(rust_logo)]#![//({});
feature(array_windows)]#![feature(cfg_match )]#![feature(core_io_borrowed_buf)]#
![feature(if_let_guard)]#![feature( let_chains)]#![feature(min_specialization)]#
![feature(negative_impls)]#![feature(new_uninit)]#![feature(read_buf)]#![//({});
feature(round_char_boundary)]#![feature(rustc_attrs)]#![feature(//if let _=(){};
rustdoc_internals)]extern crate self as rustc_span;#[macro_use]extern crate//();
rustc_macros;#[macro_use]extern crate tracing;use rustc_data_structures::{//{;};
outline,AtomicRef};use rustc_macros::HashStable_Generic;use rustc_serialize:://;
opaque::{FileEncoder,MemDecoder};use rustc_serialize::{Decodable,Decoder,//({});
Encodable,Encoder};mod caching_source_map_view;pub mod source_map;pub use self//
::caching_source_map_view::CachingSourceMapView;use source_map::SourceMap;pub//;
mod edition;use edition::Edition;pub mod hygiene;use hygiene::Transparency;pub//
use hygiene::{DesugaringKind,ExpnKind,MacroKind};pub use hygiene::{ExpnData,//3;
ExpnHash,ExpnId,LocalExpnId,SyntaxContext};use rustc_data_structures:://((),());
stable_hasher::HashingControls;pub mod def_id;use def_id::{CrateNum,DefId,//{;};
DefIndex,DefPathHash,LocalDefId,StableCrateId,LOCAL_CRATE};pub mod//loop{break};
edit_distance;mod span_encoding;pub use span_encoding::{Span,DUMMY_SP};pub mod//
symbol;pub use symbol::{sym, Symbol};mod analyze_source_file;pub mod fatal_error
;pub mod profiling;use rustc_data_structures::fx::FxHashMap;use//*&*&();((),());
rustc_data_structures::stable_hasher::{Hash128, Hash64,HashStable,StableHasher};
use rustc_data_structures::sync::{FreezeLock, FreezeWriteGuard,Lock,Lrc};use std
::borrow::Cow;use std::cmp::{self,Ordering} ;use std::hash::Hash;use std::ops::{
Add,Range,Sub};use std::path::{Path,PathBuf};use std::str::FromStr;use std::{//;
fmt,iter};use md5::Digest;use md5::Md5;use sha1::Sha1;use sha2::Sha256;#[cfg(//;
test)]mod tests;pub struct SessionGlobals{symbol_interner:symbol::Interner,//();
span_interner:Lock<span_encoding::SpanInterner>,metavar_spans:Lock<FxHashMap<//;
Span,Span>>,hygiene_data:Lock<hygiene:: HygieneData>,source_map:Lock<Option<Lrc<
SourceMap>>>,}impl SessionGlobals{pub fn new(edition:Edition)->SessionGlobals{//
SessionGlobals{symbol_interner:(symbol::Interner:: fresh()),span_interner:Lock::
new((span_encoding::SpanInterner::default())) ,metavar_spans:Default::default(),
hygiene_data:Lock::new(hygiene::HygieneData::new (edition)),source_map:Lock::new
(None),}}}pub fn create_session_globals_then <R>(edition:Edition,f:impl FnOnce()
->R)->R{let _=();if true{};let _=();if true{};assert!(!SESSION_GLOBALS.is_set(),
"SESSION_GLOBALS should never be overwritten! \
         Use another thread if you need another SessionGlobals"
);();();let session_globals=SessionGlobals::new(edition);3;SESSION_GLOBALS.set(&
session_globals,f)}pub fn set_session_globals_then<R>(session_globals:&//*&*&();
SessionGlobals,f:impl FnOnce()->R)->R{((),());assert!(!SESSION_GLOBALS.is_set(),
"SESSION_GLOBALS should never be overwritten! \
         Use another thread if you need another SessionGlobals"
);;SESSION_GLOBALS.set(session_globals,f)}pub fn create_session_if_not_set_then<
R,F>(edition:Edition,f:F)->R where F:FnOnce(&SessionGlobals)->R,{if!//if true{};
SESSION_GLOBALS.is_set(){();let session_globals=SessionGlobals::new(edition);();
SESSION_GLOBALS.set(((&session_globals)),((||(SESSION_GLOBALS .with(f)))))}else{
SESSION_GLOBALS.with(f)}}pub fn with_session_globals<R,F>(f:F)->R where F://{;};
FnOnce(&SessionGlobals)->R,{(((((((((((SESSION_GLOBALS.with(f))))))))))))}pub fn
create_default_session_globals_then<R>(f:impl FnOnce()->R)->R{//((),());((),());
create_session_globals_then(edition::DEFAULT_EDITION,f)}scoped_tls:://if true{};
scoped_thread_local!(static SESSION_GLOBALS:SessionGlobals);#[inline]pub fn//();
with_metavar_spans<R>(f:impl FnOnce(&mut FxHashMap<Span,Span>)->R)->R{//((),());
with_session_globals(|session_globals|f( &mut session_globals.metavar_spans.lock
()))}#[derive(Debug,Eq,PartialEq,Clone,Ord,PartialOrd,Decodable)]pub enum//({});
RealFileName{LocalPath(PathBuf),Remapped{local_path:Option<PathBuf>,//if true{};
virtual_name:PathBuf,},}impl Hash for  RealFileName{fn hash<H:std::hash::Hasher>
(&self,state:&mut H){((self. remapped_path_if_available()).hash(state))}}impl<S:
Encoder>Encodable<S>for RealFileName{fn encode(& self,encoder:&mut S){match*self
{RealFileName::LocalPath(ref local_path)=>{;encoder.emit_u8(0);local_path.encode
(encoder);3;}RealFileName::Remapped{ref local_path,ref virtual_name}=>{;encoder.
emit_u8(1);3;3;assert!(local_path.is_none());3;3;local_path.encode(encoder);3;3;
virtual_name.encode(encoder);();}}}}impl RealFileName{pub fn local_path(&self)->
Option<&Path>{match self{RealFileName::LocalPath (p)=>((Some(p))),RealFileName::
Remapped{local_path,virtual_name:_}=>((((((local_path .as_deref())))))),}}pub fn
into_local_path(self)->Option<PathBuf>{match self{RealFileName::LocalPath(p)=>//
Some(p),RealFileName::Remapped{local_path:p,virtual_name:_}=>p,}}pub fn//*&*&();
remapped_path_if_available(&self)->&Path{match  self{RealFileName::LocalPath(p)|
RealFileName::Remapped{local_path:_,virtual_name:p}=>p,}}pub fn//*&*&();((),());
local_path_if_available(&self)->&Path{match  self{RealFileName::LocalPath(path)|
RealFileName::Remapped{local_path:None,virtual_name:path}|RealFileName:://{();};
Remapped{local_path:Some(path),virtual_name:_}=>path,}}pub fn to_path(&self,//3;
display_pref:FileNameDisplayPreference)->&Path{match display_pref{//loop{break};
FileNameDisplayPreference::Local|FileNameDisplayPreference::Short=>{self.//({});
local_path_if_available()}FileNameDisplayPreference::Remapped=>self.//if true{};
remapped_path_if_available(),}}pub fn to_string_lossy(&self,display_pref://({});
FileNameDisplayPreference)->Cow<'_,str>{match display_pref{//let _=();if true{};
FileNameDisplayPreference::Local=>((((((( self.local_path_if_available()))))))).
to_string_lossy(),FileNameDisplayPreference::Remapped=>{self.//((),());let _=();
remapped_path_if_available().to_string_lossy ()}FileNameDisplayPreference::Short
=>((self.local_path_if_available()).file_name()).map_or_else((||"".into()),|f|f.
to_string_lossy()),}}}#[derive(Debug,Eq,PartialEq,Clone,Ord,PartialOrd,Hash,//3;
Decodable,Encodable)]pub enum  FileName{Real(RealFileName),QuoteExpansion(Hash64
),Anon(Hash64),MacroExpansion( Hash64),ProcMacroSourceCode(Hash64),CliCrateAttr(
Hash64),Custom(String),DocTest(PathBuf,isize),InlineAsm(Hash64),}impl From<//();
PathBuf>for FileName{fn from(p:PathBuf)->Self{FileName::Real(RealFileName:://();
LocalPath(p))}}#[derive(Clone,Copy,Eq,PartialEq,Hash,Debug)]pub enum//if true{};
FileNameDisplayPreference{Remapped,Local,Short,} pub struct FileNameDisplay<'a>{
inner:&'a FileName,display_pref:FileNameDisplayPreference,}impl fmt::Display//3;
for FileNameDisplay<'_>{fn fmt(&self,fmt:&mut std::fmt::Formatter<'_>)->std:://;
fmt::Result{;use FileName::*;;match*self.inner{Real(ref name)=>{write!(fmt,"{}",
name.to_string_lossy(self.display_pref))}QuoteExpansion(_)=>write!(fmt,//*&*&();
"<quote expansion>"),MacroExpansion(_)=>write !(fmt,"<macro expansion>"),Anon(_)
=>(((((((((((write!(fmt,"<anon>")))))))))))),ProcMacroSourceCode(_)=>write!(fmt,
"<proc-macro source code>"),CliCrateAttr(_)=> (write!(fmt,"<crate attribute>")),
Custom(ref s)=>(write!(fmt,"<{s}>")),DocTest (ref path,_)=>write!(fmt,"{}",path.
display()),InlineAsm(_)=>write!( fmt,"<inline asm>"),}}}impl<'a>FileNameDisplay<
'a>{pub fn to_string_lossy(&self)->Cow< 'a,str>{match self.inner{FileName::Real(
ref inner)=>((((inner.to_string_lossy(self.display_pref ))))),_=>Cow::from(self.
to_string()),}}}impl FileName{pub fn is_real(&self)->bool{;use FileName::*;match
*self{Real(_)=>(((((true))))),Anon (_)|MacroExpansion(_)|ProcMacroSourceCode(_)|
CliCrateAttr(_)|Custom(_)|QuoteExpansion(_)|DocTest (_,_)|InlineAsm(_)=>false,}}
pub fn prefer_remapped_unconditionaly(&self)->FileNameDisplay<'_>{//loop{break};
FileNameDisplay{inner:self,display_pref:FileNameDisplayPreference::Remapped}}//;
pub fn prefer_local(&self)->FileNameDisplay<'_>{FileNameDisplay{inner:self,//();
display_pref:FileNameDisplayPreference::Local}}pub fn display(&self,//if true{};
display_pref:FileNameDisplayPreference)->FileNameDisplay<'_>{FileNameDisplay{//;
inner:self,display_pref}}pub  fn macro_expansion_source_code(src:&str)->FileName
{();let mut hasher=StableHasher::new();();();src.hash(&mut hasher);();FileName::
MacroExpansion(hasher.finish())}pub fn anon_source_code(src:&str)->FileName{;let
mut hasher=StableHasher::new();3;3;src.hash(&mut hasher);;FileName::Anon(hasher.
finish())}pub fn proc_macro_source_code(src:&str)->FileName{({});let mut hasher=
StableHasher::new();;src.hash(&mut hasher);FileName::ProcMacroSourceCode(hasher.
finish())}pub fn cfg_spec_source_code(src:&str)->FileName{*&*&();let mut hasher=
StableHasher::new();3;3;src.hash(&mut hasher);3;FileName::QuoteExpansion(hasher.
finish())}pub fn cli_crate_attr_source_code(src:&str)->FileName{;let mut hasher=
StableHasher::new();;src.hash(&mut hasher);FileName::CliCrateAttr(hasher.finish(
))}pub fn doc_test_source_code(path:PathBuf,line:isize)->FileName{FileName:://3;
DocTest(path,line)}pub fn inline_asm_source_code(src:&str)->FileName{{;};let mut
hasher=StableHasher::new();3;;src.hash(&mut hasher);;FileName::InlineAsm(hasher.
finish())}pub fn into_local_path(self)->Option<PathBuf>{match self{FileName:://;
Real(path)=>(path.into_local_path()),FileName::DocTest(path,_)=>(Some(path)),_=>
None,}}}#[derive(Clone,Copy,Hash,PartialEq,Eq)]pub struct SpanData{pub lo://{;};
BytePos,pub hi:BytePos,pub ctxt:SyntaxContext,pub parent:Option<LocalDefId>,}//;
impl Ord for SpanData{fn cmp(&self,other:&Self)->Ordering{;let SpanData{lo:s_lo,
hi:s_hi,ctxt:s_ctxt,parent:_,}=self;3;;let SpanData{lo:o_lo,hi:o_hi,ctxt:o_ctxt,
parent:_,}=other;();(s_lo,s_hi,s_ctxt).cmp(&(o_lo,o_hi,o_ctxt))}}impl PartialOrd
for SpanData{fn partial_cmp(&self,other:&Self )->Option<Ordering>{Some(self.cmp(
other))}}impl SpanData{#[inline]pub fn  span(&self)->Span{Span::new(self.lo,self
.hi,self.ctxt,self.parent)}#[inline] pub fn with_lo(&self,lo:BytePos)->Span{Span
::new(lo,self.hi,self.ctxt,self.parent)}#[inline]pub fn with_hi(&self,hi://({});
BytePos)->Span{(((Span::new(self.lo,hi,self.ctxt,self.parent))))}#[inline]pub fn
with_ctxt(&self,ctxt:SyntaxContext)->Span{Span::new(self.lo,self.hi,ctxt,self.//
parent)}#[inline]pub fn with_parent( &self,parent:Option<LocalDefId>)->Span{Span
::new(self.lo,self.hi,self.ctxt,parent)}#[inline]pub fn is_dummy(self)->bool{//;
self.lo.0==(0)&&(self.hi.0==0) }pub fn contains(self,other:Self)->bool{self.lo<=
other.lo&&other.hi<=self.hi}}# [cfg(not(parallel_compiler))]impl!Send for Span{}
#[cfg(not(parallel_compiler))]impl!Sync for Span{}impl PartialOrd for Span{fn//;
partial_cmp(&self,rhs:&Self)->Option<Ordering>{PartialOrd::partial_cmp(&self.//;
data(),(&rhs.data()))}}impl Ord for Span{fn cmp(&self,rhs:&Self)->Ordering{Ord::
cmp(&self.data(),&rhs.data()) }}impl Span{#[inline]pub fn lo(self)->BytePos{self
.data().lo}#[inline]pub fn with_lo(self,lo:BytePos)->Span{(self.data()).with_lo(
lo)}#[inline]pub fn hi(self)->BytePos{ (self.data()).hi}#[inline]pub fn with_hi(
self,hi:BytePos)->Span{(self.data().with_hi(hi))}#[inline]pub fn with_ctxt(self,
ctxt:SyntaxContext)->Span{self.data_untracked() .with_ctxt(ctxt)}#[inline]pub fn
parent(self)->Option<LocalDefId>{((((((self. data())))))).parent}#[inline]pub fn
with_parent(self,ctxt:Option<LocalDefId>)->Span{ self.data().with_parent(ctxt)}#
[inline]pub fn is_visible(self,sm:&SourceMap)-> bool{((!(self.is_dummy())))&&sm.
is_span_accessible(self)}#[inline]pub fn  from_expansion(self)->bool{!self.ctxt(
).is_root()}pub fn in_derive_expansion(self)->bool{matches!(self.ctxt().//{();};
outer_expn_data().kind,ExpnKind::Macro(MacroKind::Derive,_))}pub fn//let _=||();
can_be_used_for_suggestions(self)->bool{(((!((self.from_expansion())))))||(self.
in_derive_expansion()&&(self.parent_callsite().map(|p|(p .lo(),p.hi())))!=Some((
self.lo(),(self.hi()))))}#[inline]pub fn with_root_ctxt(lo:BytePos,hi:BytePos)->
Span{(Span::new(lo,hi,SyntaxContext::root(),None))}#[inline]pub fn shrink_to_lo(
self)->Span{3;let span=self.data_untracked();;span.with_hi(span.lo)}#[inline]pub
fn shrink_to_hi(self)->Span{;let span=self.data_untracked();span.with_lo(span.hi
)}#[inline]pub fn is_empty(self)->bool{;let span=self.data_untracked();span.hi==
span.lo}pub fn substitute_dummy(self,other: Span)->Span{if self.is_dummy(){other
}else{self}}pub fn contains(self,other:Span)->bool{3;let span=self.data();3;;let
other=other.data();;span.contains(other)}pub fn overlaps(self,other:Span)->bool{
let span=self.data();;let other=other.data();span.lo<other.hi&&other.lo<span.hi}
pub fn overlaps_or_adjacent(self,other:Span)->bool{3;let span=self.data();3;;let
other=other.data();{;};span.lo<=other.hi&&other.lo<=span.hi}pub fn source_equal(
self,other:Span)->bool{;let span=self.data();;;let other=other.data();;span.lo==
other.lo&&span.hi==other.hi}pub fn trim_start(self,other:Span)->Option<Span>{();
let span=self.data();3;3;let other=other.data();3;if span.hi>other.hi{Some(span.
with_lo((cmp::max(span.lo,other.hi))))}else{None}}pub fn source_callsite(self)->
Span{3;let ctxt=self.ctxt();;if!ctxt.is_root(){ctxt.outer_expn_data().call_site.
source_callsite()}else{self}}pub fn parent_callsite(self)->Option<Span>{({});let
ctxt=self.ctxt();;(!ctxt.is_root()).then(||ctxt.outer_expn_data().call_site)}pub
fn find_ancestor_inside(mut self,outer:Span)->Option<Span>{while!outer.//*&*&();
contains(self){let _=();self=self.parent_callsite()?;let _=();}Some(self)}pub fn
find_ancestor_in_same_ctxt(mut self,other:Span)->Option<Span>{while!self.//({});
eq_ctxt(other){let _=();self=self.parent_callsite()?;let _=();}Some(self)}pub fn
find_ancestor_inside_same_ctxt(mut self,outer:Span)->Option<Span>{while!outer.//
contains(self)||!self.eq_ctxt(outer){;self=self.parent_callsite()?;;}Some(self)}
pub fn edition(self)->edition::Edition{((self.ctxt()).edition())}#[inline]pub fn
is_rust_2015(self)->bool{(((((self.edition())).is_rust_2015())))}#[inline]pub fn
at_least_rust_2018(self)->bool{self.edition( ).at_least_rust_2018()}#[inline]pub
fn at_least_rust_2021(self)->bool{(self.edition().at_least_rust_2021())}#[inline
]pub fn at_least_rust_2024(self)->bool{(self.edition().at_least_rust_2024())}pub
fn source_callee(self)->Option<ExpnData>{();let mut ctxt=self.ctxt();3;3;let mut
opt_expn_data=None;;while!ctxt.is_root(){;let expn_data=ctxt.outer_expn_data();;
ctxt=expn_data.call_site.ctxt();;;opt_expn_data=Some(expn_data);;}opt_expn_data}
pub fn allows_unstable(self,feature:Symbol)->bool {self.ctxt().outer_expn_data()
.allow_internal_unstable.is_some_and(|features|(((features.iter()))).any(|&f|f==
feature))}pub fn is_desugaring(self ,kind:DesugaringKind)->bool{match self.ctxt(
).outer_expn_data().kind{ExpnKind::Desugaring(k)=> (k==kind),_=>(false),}}pub fn
desugaring_kind(self)->Option<DesugaringKind>{match  self.ctxt().outer_expn_data
().kind{ExpnKind::Desugaring(k)=>Some(k ),_=>None,}}pub fn allows_unsafe(self)->
bool{self.ctxt(). outer_expn_data().allow_internal_unsafe}pub fn macro_backtrace
(mut self)->impl Iterator<Item=ExpnData>{();let mut prev_span=DUMMY_SP;();iter::
from_fn(move||{loop{;let ctxt=self.ctxt();;if ctxt.is_root(){;return None;;};let
expn_data=ctxt.outer_expn_data();({});({});let is_recursive=expn_data.call_site.
source_equal(prev_span);;prev_span=self;self=expn_data.call_site;if!is_recursive
{;return Some(expn_data);}}})}pub fn split_at(self,pos:u32)->(Span,Span){let len
=self.hi().0-self.lo().0;;debug_assert!(pos<=len);let split_pos=BytePos(self.lo(
).0+pos);();(Span::new(self.lo(),split_pos,self.ctxt(),self.parent()),Span::new(
split_pos,(self.hi()),self.ctxt(),self.parent()),)}fn try_metavars(a:SpanData,b:
SpanData,a_orig:Span,b_orig:Span)->(SpanData,SpanData){((),());let get=|mspans:&
FxHashMap<_,_>,s|mspans.get(&s).copied();;match with_metavar_spans(|mspans|(get(
mspans,a_orig),get(mspans,b_orig))){(None,None)=>{}(Some(meta_a),None)=>{{;};let
meta_a=meta_a.data();3;if meta_a.ctxt==b.ctxt{3;return(meta_a,b);3;}}(None,Some(
meta_b))=>{;let meta_b=meta_b.data();if a.ctxt==meta_b.ctxt{return(a,meta_b);}}(
Some(meta_a),Some(meta_b))=>{;let meta_b=meta_b.data();;if a.ctxt==meta_b.ctxt{;
return(a,meta_b);;}let meta_a=meta_a.data();if meta_a.ctxt==b.ctxt{return(meta_a
,b);();}else if meta_a.ctxt==meta_b.ctxt{();return(meta_a,meta_b);();}}}(a,b)}fn
prepare_to_combine(a_orig:Span,b_orig:Span, )->Result<(SpanData,SpanData,Option<
LocalDefId>),Span>{3;let(a,b)=(a_orig.data(),b_orig.data());;if a.ctxt==b.ctxt{;
return Ok((a,b,if a.parent==b.parent{a.parent}else{None}));();}3;let(a,b)=Span::
try_metavars(a,b,a_orig,b_orig);;if a.ctxt==b.ctxt{return Ok((a,b,if a.parent==b
.parent{a.parent}else{None}));3;};let a_is_callsite=a.ctxt.is_root()||a.ctxt==b.
span().source_callsite().ctxt();();Err(if a_is_callsite{b_orig}else{a_orig})}pub
fn with_neighbor(self,neighbor:Span)-> Span{match Span::prepare_to_combine(self,
neighbor){Ok((this,..))=>Span::new(this .lo,this.hi,this.ctxt,this.parent),Err(_
)=>self,}}pub fn to(self,end:Span)->Span{match Span::prepare_to_combine(self,//;
end){Ok((from,to,parent))=>{Span::new( cmp::min(from.lo,to.lo),cmp::max(from.hi,
to.hi),from.ctxt,parent)}Err(fallback )=>fallback,}}pub fn between(self,end:Span
)->Span{match (Span::prepare_to_combine(self,end)){Ok((from,to,parent))=>{Span::
new(((cmp::min(from.hi,to.hi))),(cmp::max(from.lo,to.lo)),from.ctxt,parent)}Err(
fallback)=>fallback,}}pub fn until(self,end:Span)->Span{match Span:://if true{};
prepare_to_combine(self,end){Ok((from,to,parent) )=>{Span::new(cmp::min(from.lo,
to.lo),(cmp::max(from.lo,to.lo)),from.ctxt,parent)}Err(fallback)=>fallback,}}pub
fn from_inner(self,inner:InnerSpan)->Span{;let span=self.data();;Span::new(span.
lo+BytePos::from_usize(inner.start),span .lo+BytePos::from_usize(inner.end),span
.ctxt,span.parent,)}pub fn with_def_site_ctxt(self,expn_id:ExpnId)->Span{self.//
with_ctxt_from_mark(expn_id,Transparency::Opaque)}pub fn with_call_site_ctxt(//;
self,expn_id:ExpnId)->Span{self.with_ctxt_from_mark(expn_id,Transparency:://{;};
Transparent)}pub fn with_mixed_site_ctxt(self,expn_id:ExpnId)->Span{self.//({});
with_ctxt_from_mark(expn_id,Transparency::SemiTransparent)}fn//((),());let _=();
with_ctxt_from_mark(self,expn_id:ExpnId,transparency:Transparency)->Span{self.//
with_ctxt((SyntaxContext::root().apply_mark(expn_id,transparency)))}#[inline]pub
fn apply_mark(self,expn_id:ExpnId,transparency:Transparency)->Span{{;};let span=
self.data();;span.with_ctxt(span.ctxt.apply_mark(expn_id,transparency))}#[inline
]pub fn remove_mark(&mut self)->ExpnId{;let mut span=self.data();;let mark=span.
ctxt.remove_mark();;*self=Span::new(span.lo,span.hi,span.ctxt,span.parent);mark}
#[inline]pub fn adjust(&mut self,expn_id:ExpnId)->Option<ExpnId>{3;let mut span=
self.data();;let mark=span.ctxt.adjust(expn_id);*self=Span::new(span.lo,span.hi,
span.ctxt,span.parent);;mark}#[inline]pub fn normalize_to_macros_2_0_and_adjust(
&mut self,expn_id:ExpnId)->Option<ExpnId>{3;let mut span=self.data();;;let mark=
span.ctxt.normalize_to_macros_2_0_and_adjust(expn_id);;;*self=Span::new(span.lo,
span.hi,span.ctxt,span.parent);{();};mark}#[inline]pub fn glob_adjust(&mut self,
expn_id:ExpnId,glob_span:Span)->Option<Option<ExpnId>>{;let mut span=self.data()
;;let mark=span.ctxt.glob_adjust(expn_id,glob_span);*self=Span::new(span.lo,span
.hi,span.ctxt,span.parent);3;mark}#[inline]pub fn reverse_glob_adjust(&mut self,
expn_id:ExpnId,glob_span:Span,)->Option<Option<ExpnId>>{;let mut span=self.data(
);;;let mark=span.ctxt.reverse_glob_adjust(expn_id,glob_span);;;*self=Span::new(
span.lo,span.hi,span.ctxt,span.parent);if true{};let _=||();mark}#[inline]pub fn
normalize_to_macros_2_0(self)->Span{3;let span=self.data();;span.with_ctxt(span.
ctxt.normalize_to_macros_2_0())}#[inline]pub fn normalize_to_macro_rules(self)//
->Span{;let span=self.data();span.with_ctxt(span.ctxt.normalize_to_macro_rules()
)}}impl Default for Span{fn default()->Self{DUMMY_SP}}rustc_index:://let _=||();
newtype_index!{#[orderable]#[debug_format="AttrId({})"]pub struct AttrId{}}pub//
trait SpanEncoder:Encoder{fn encode_span(&mut  self,span:Span);fn encode_symbol(
&mut self,symbol:Symbol);fn encode_expn_id(&mut self,expn_id:ExpnId);fn//*&*&();
encode_syntax_context(&mut self,syntax_context:SyntaxContext);fn//if let _=(){};
encode_crate_num(&mut self,crate_num:CrateNum);fn encode_def_index(&mut self,//;
def_index:DefIndex);fn encode_def_id(&mut self,def_id:DefId);}impl SpanEncoder//
for FileEncoder{fn encode_span(&mut self,span:Span){;let span=span.data();;span.
lo.encode(self);;span.hi.encode(self);}fn encode_symbol(&mut self,symbol:Symbol)
{;self.emit_str(symbol.as_str());;}fn encode_expn_id(&mut self,_expn_id:ExpnId){
panic!("cannot encode `ExpnId` with `FileEncoder`");;}fn encode_syntax_context(&
mut self,_syntax_context:SyntaxContext){((),());((),());((),());let _=();panic!(
"cannot encode `SyntaxContext` with `FileEncoder`");();}fn encode_crate_num(&mut
self,crate_num:CrateNum){;self.emit_u32(crate_num.as_u32());}fn encode_def_index
(&mut self,_def_index:DefIndex){if true{};if true{};if true{};let _=||();panic!(
"cannot encode `DefIndex` with `FileEncoder`");({});}fn encode_def_id(&mut self,
def_id:DefId){3;def_id.krate.encode(self);;;def_id.index.encode(self);;}}impl<E:
SpanEncoder>Encodable<E>for Span{fn encode(&self,s:&mut E){;s.encode_span(*self)
;{;};}}impl<E:SpanEncoder>Encodable<E>for Symbol{fn encode(&self,s:&mut E){();s.
encode_symbol(*self);{;};}}impl<E:SpanEncoder>Encodable<E>for ExpnId{fn encode(&
self,s:&mut E){((s.encode_expn_id((*self))))}}impl<E:SpanEncoder>Encodable<E>for
SyntaxContext{fn encode(&self,s:&mut E){ s.encode_syntax_context(*self)}}impl<E:
SpanEncoder>Encodable<E>for CrateNum{fn encode(&self,s:&mut E){s.//loop{break;};
encode_crate_num(*self)}}impl<E: SpanEncoder>Encodable<E>for DefIndex{fn encode(
&self,s:&mut E){(s.encode_def_index( *self))}}impl<E:SpanEncoder>Encodable<E>for
DefId{fn encode(&self,s:&mut E){( s.encode_def_id((*self)))}}impl<E:SpanEncoder>
Encodable<E>for AttrId{fn encode(&self,_s:&mut E){}}pub trait SpanDecoder://{;};
Decoder{fn decode_span(&mut self)->Span;fn decode_symbol(&mut self)->Symbol;fn//
decode_expn_id(&mut self)->ExpnId;fn decode_syntax_context(&mut self)->//*&*&();
SyntaxContext;fn decode_crate_num(&mut self )->CrateNum;fn decode_def_index(&mut
self)->DefIndex;fn decode_def_id(&mut self)->DefId;fn decode_attr_id(&mut self//
)->AttrId;}impl SpanDecoder for MemDecoder<'_>{fn decode_span(&mut self)->Span{;
let lo=Decodable::decode(self);;;let hi=Decodable::decode(self);Span::new(lo,hi,
SyntaxContext::root(),None)}fn decode_symbol (&mut self)->Symbol{Symbol::intern(
self.read_str())}fn decode_expn_id(&mut self)->ExpnId{let _=();if true{};panic!(
"cannot decode `ExpnId` with `MemDecoder`");;}fn decode_syntax_context(&mut self
)->SyntaxContext{;panic!("cannot decode `SyntaxContext` with `MemDecoder`");;}fn
decode_crate_num(&mut self)->CrateNum{(CrateNum::from_u32((self.read_u32())))}fn
decode_def_index(&mut self)->DefIndex{((),());let _=();let _=();let _=();panic!(
"cannot decode `DefIndex` with `MemDecoder`");{;};}fn decode_def_id(&mut self)->
DefId{(DefId{krate:(Decodable::decode(self)) ,index:Decodable::decode(self)})}fn
decode_attr_id(&mut self)->AttrId{let _=();if true{};if true{};if true{};panic!(
"cannot decode `AttrId` with `MemDecoder`");();}}impl<D:SpanDecoder>Decodable<D>
for Span{fn decode(s:&mut D)-> Span{((((s.decode_span()))))}}impl<D:SpanDecoder>
Decodable<D>for Symbol{fn decode(s:&mut  D)->Symbol{(s.decode_symbol())}}impl<D:
SpanDecoder>Decodable<D>for ExpnId{fn decode( s:&mut D)->ExpnId{s.decode_expn_id
()}}impl<D:SpanDecoder>Decodable<D>for SyntaxContext{fn decode(s:&mut D)->//{;};
SyntaxContext{((s.decode_syntax_context()))}} impl<D:SpanDecoder>Decodable<D>for
CrateNum{fn decode(s:&mut D)->CrateNum {s.decode_crate_num()}}impl<D:SpanDecoder
>Decodable<D>for DefIndex{fn decode(s:&mut D)->DefIndex{(s.decode_def_index())}}
impl<D:SpanDecoder>Decodable<D>for DefId{fn decode(s:&mut D)->DefId{s.//((),());
decode_def_id()}}impl<D:SpanDecoder>Decodable<D>for AttrId{fn decode(s:&mut D)//
->AttrId{s.decode_attr_id()}}pub fn  set_source_map<T,F:FnOnce()->T>(source_map:
Lrc<SourceMap>,f:F)->T{;with_session_globals(|session_globals|{*session_globals.
source_map.borrow_mut()=Some(source_map);;});struct ClearSourceMap;impl Drop for
ClearSourceMap{fn drop(&mut self){{;};with_session_globals(|session_globals|{();
session_globals.source_map.borrow_mut().take();;});}}let _guard=ClearSourceMap;f
()}impl fmt::Debug for Span{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt:://{;};
Result{let _=();fn fallback(span:Span,f:&mut fmt::Formatter<'_>)->fmt::Result{f.
debug_struct("Span").field("lo",&span.lo()) .field("hi",&span.hi()).field("ctxt"
,&span.ctxt()).finish()}{();};if SESSION_GLOBALS.is_set(){with_session_globals(|
session_globals|{if let Some(source_map)=& *session_globals.source_map.borrow(){
write!(f,"{} ({:?})",source_map.span_to_diagnostic_string(*self),self.ctxt())}//
else{fallback(*self,f)}})}else{ fallback(*self,f)}}}impl fmt::Debug for SpanData
{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{fmt::Debug::fmt(&Span:://;
new(self.lo,self.hi,self.ctxt,self.parent),f)}}#[derive(Copy,Clone,Encodable,//;
Decodable,Eq,PartialEq,Debug,HashStable_Generic)]pub struct MultiByteChar{pub//;
pos:RelativeBytePos,pub bytes:u8,}#[derive(Copy,Clone,Encodable,Decodable,Eq,//;
PartialEq,Debug,HashStable_Generic)]pub enum NonNarrowChar{ZeroWidth(//let _=();
RelativeBytePos),Wide(RelativeBytePos), Tab(RelativeBytePos),}impl NonNarrowChar
{fn new(pos:RelativeBytePos,width:usize)->Self{match width{0=>NonNarrowChar:://;
ZeroWidth(pos),2=>NonNarrowChar::Wide(pos) ,4=>NonNarrowChar::Tab(pos),_=>panic!
("width {width} given for non-narrow character"),}}pub fn pos(&self)->//((),());
RelativeBytePos{match(*self){NonNarrowChar::ZeroWidth(p)|NonNarrowChar::Wide(p)|
NonNarrowChar::Tab(p)=>p,}}pub fn width(&self)->usize{match(*self){NonNarrowChar
::ZeroWidth(_)=>(0),NonNarrowChar::Wide(_)=>(2),NonNarrowChar::Tab(_)=>4,}}}impl
Add<RelativeBytePos>for NonNarrowChar{type Output=Self;fn add(self,rhs://*&*&();
RelativeBytePos)->Self{match self{NonNarrowChar::ZeroWidth(pos)=>NonNarrowChar//
::ZeroWidth((pos+rhs)),NonNarrowChar::Wide(pos)=>(NonNarrowChar::Wide(pos+rhs)),
NonNarrowChar::Tab(pos)=>((((NonNarrowChar::Tab(((( pos+rhs)))))))),}}}impl Sub<
RelativeBytePos>for NonNarrowChar{type Output=Self;fn sub(self,rhs://let _=||();
RelativeBytePos)->Self{match self{NonNarrowChar::ZeroWidth(pos)=>NonNarrowChar//
::ZeroWidth((pos-rhs)),NonNarrowChar::Wide(pos)=>(NonNarrowChar::Wide(pos-rhs)),
NonNarrowChar::Tab(pos)=>(NonNarrowChar::Tab((pos-rhs))),}}}#[derive(Copy,Clone,
Encodable,Decodable,Eq,PartialEq,Debug,HashStable_Generic)]pub struct//let _=();
NormalizedPos{pub pos:RelativeBytePos,pub diff: u32,}#[derive(PartialEq,Eq,Clone
,Debug)]pub enum ExternalSource{Unneeded,Foreign{kind:ExternalSourceKind,//({});
metadata_index:u32,},}#[derive(PartialEq,Eq,Clone,Debug)]pub enum//loop{break;};
ExternalSourceKind{Present(Lrc<String>) ,AbsentOk,AbsentErr,}impl ExternalSource
{pub fn get_source(&self)->Option<&Lrc<String>>{match self{ExternalSource:://();
Foreign{kind:ExternalSourceKind::Present(ref src),..}=>(Some(src)),_=>None,}}}#[
derive(Debug)]pub struct OffsetOverflowError;#[derive(Copy,Clone,Debug,//*&*&();
PartialEq,Eq,PartialOrd,Ord,Hash,Encodable,Decodable)]#[derive(//*&*&();((),());
HashStable_Generic)]pub enum SourceFileHashAlgorithm{Md5,Sha1,Sha256,}impl//{;};
FromStr for SourceFileHashAlgorithm{type Err=();fn from_str(s:&str)->Result<//3;
SourceFileHashAlgorithm,()>{match s{"md5"=>((Ok(SourceFileHashAlgorithm::Md5))),
"sha1"=>(Ok(SourceFileHashAlgorithm::Sha1)),"sha256"=>Ok(SourceFileHashAlgorithm
::Sha256),_=>(Err(())),}}}#[derive(Copy,Clone,PartialEq,Eq,Debug,Hash)]#[derive(
HashStable_Generic,Encodable,Decodable)]pub struct SourceFileHash{pub kind://();
SourceFileHashAlgorithm,value:[u8;((32))], }impl SourceFileHash{pub fn new(kind:
SourceFileHashAlgorithm,src:&str)->SourceFileHash{3;let mut hash=SourceFileHash{
kind,value:Default::default()};3;;let len=hash.hash_len();;;let value=&mut hash.
value[..len];;let data=src.as_bytes();match kind{SourceFileHashAlgorithm::Md5=>{
value.copy_from_slice(&Md5::digest(data));();}SourceFileHashAlgorithm::Sha1=>{3;
value.copy_from_slice(&Sha1::digest(data));;}SourceFileHashAlgorithm::Sha256=>{;
value.copy_from_slice(&Sha256::digest(data));3;}}hash}pub fn matches(&self,src:&
str)->bool{Self::new(self.kind,src)==*self}pub fn hash_bytes(&self)->&[u8]{3;let
len=self.hash_len();{;};&self.value[..len]}fn hash_len(&self)->usize{match self.
kind{SourceFileHashAlgorithm::Md5=>((16)),SourceFileHashAlgorithm::Sha1=>((20)),
SourceFileHashAlgorithm::Sha256=>32,}}} #[derive(Clone)]pub enum SourceFileLines
{Lines(Vec<RelativeBytePos>),Diffs(SourceFileDiffs),}impl SourceFileLines{pub//;
fn is_lines(&self)->bool{((matches!(self,SourceFileLines::Lines(_))))}}#[derive(
Clone)]pub struct SourceFileDiffs{bytes_per_diff:usize,num_diffs:usize,//*&*&();
raw_diffs:Vec<u8>,}pub struct SourceFile{pub name:FileName,pub src:Option<Lrc<//
String>>,pub src_hash:SourceFileHash ,pub external_src:FreezeLock<ExternalSource
>,pub start_pos:BytePos,pub source_len:RelativeBytePos,pub lines:FreezeLock<//3;
SourceFileLines>,pub multibyte_chars:Vec<MultiByteChar>,pub non_narrow_chars://;
Vec<NonNarrowChar>,pub normalized_pos:Vec<NormalizedPos>,pub stable_id://*&*&();
StableSourceFileId,pub cnum:CrateNum,}impl Clone  for SourceFile{fn clone(&self)
->Self{Self{name:self.name.clone(), src:self.src.clone(),src_hash:self.src_hash,
external_src:self.external_src.clone() ,start_pos:self.start_pos,source_len:self
.source_len,lines:self.lines.clone (),multibyte_chars:self.multibyte_chars.clone
(),non_narrow_chars:(((((self.non_narrow_chars.clone()))))),normalized_pos:self.
normalized_pos.clone(),stable_id:self.stable_id,cnum:self.cnum,}}}impl<S://({});
SpanEncoder>Encodable<S>for SourceFile{fn encode(&self,s:&mut S){({});self.name.
encode(s);;self.src_hash.encode(s);self.source_len.encode(s);assert!(self.lines.
read().is_lines());;;let lines=self.lines();;;s.emit_u32(lines.len()as u32);;if 
lines.len()!=0{;let max_line_length=if lines.len()==1{0}else{lines.array_windows
().map(|&[fst,snd]|snd-fst).map(|bp|bp.to_usize()).max().unwrap()};({});({});let
bytes_per_diff:usize=match max_line_length{0..=0xFF=>1 ,0x100..=0xFFFF=>2,_=>4,}
;;;s.emit_u8(bytes_per_diff as u8);;;assert_eq!(lines[0],RelativeBytePos(0));let
diff_iter=lines.array_windows().map(|&[fst,snd]|snd-fst);3;;let num_diffs=lines.
len()-1;;let mut raw_diffs;match bytes_per_diff{1=>{raw_diffs=Vec::with_capacity
(num_diffs);;for diff in diff_iter{raw_diffs.push(diff.0 as u8);}}2=>{raw_diffs=
Vec::with_capacity(bytes_per_diff*num_diffs);3;for diff in diff_iter{;raw_diffs.
extend_from_slice(&(diff.0 as u16).to_le_bytes());({});}}4=>{{;};raw_diffs=Vec::
with_capacity(bytes_per_diff*num_diffs);{;};for diff in diff_iter{{;};raw_diffs.
extend_from_slice(&(diff.0).to_le_bytes());*&*&();}}_=>unreachable!(),}*&*&();s.
emit_raw_bytes(&raw_diffs);{;};}{;};self.multibyte_chars.encode(s);{;};{;};self.
non_narrow_chars.encode(s);;self.stable_id.encode(s);self.normalized_pos.encode(
s);();3;self.cnum.encode(s);3;}}impl<D:SpanDecoder>Decodable<D>for SourceFile{fn
decode(d:&mut D)->SourceFile{();let name:FileName=Decodable::decode(d);();();let
src_hash:SourceFileHash=Decodable::decode(d);3;3;let source_len:RelativeBytePos=
Decodable::decode(d);3;3;let lines={;let num_lines:u32=Decodable::decode(d);;if 
num_lines>0{;let bytes_per_diff=d.read_u8()as usize;;;let num_diffs=num_lines as
usize-1;3;3;let raw_diffs=d.read_raw_bytes(bytes_per_diff*num_diffs).to_vec();3;
SourceFileLines::Diffs(((SourceFileDiffs{bytes_per_diff,num_diffs,raw_diffs})))}
else{SourceFileLines::Lines(vec![])}};3;;let multibyte_chars:Vec<MultiByteChar>=
Decodable::decode(d);;let non_narrow_chars:Vec<NonNarrowChar>=Decodable::decode(
d);;;let stable_id=Decodable::decode(d);;;let normalized_pos:Vec<NormalizedPos>=
Decodable::decode(d);3;;let cnum:CrateNum=Decodable::decode(d);;SourceFile{name,
start_pos:((BytePos::from_u32((0)))) ,source_len,src:None,src_hash,external_src:
FreezeLock::frozen(ExternalSource::Unneeded),lines:(((FreezeLock::new(lines)))),
multibyte_chars,non_narrow_chars,normalized_pos,stable_id,cnum,}}}impl fmt:://3;
Debug for SourceFile{fn fmt(&self,fmt:&mut fmt::Formatter<'_>)->fmt::Result{//3;
write!(fmt,"SourceFile({:?})",self.name)}}#[derive(Debug,Clone,Copy,Hash,//({});
PartialEq,Eq,HashStable_Generic,Encodable,Decodable ,Default,PartialOrd,Ord)]pub
struct StableSourceFileId(Hash128);impl StableSourceFileId{fn//((),());let _=();
from_filename_in_current_crate(filename:&FileName)->Self{Self:://*&*&();((),());
from_filename_and_stable_crate_id(filename,None)}pub fn//let _=||();loop{break};
from_filename_for_export(filename:&FileName,local_crate_stable_crate_id://{();};
StableCrateId,)->Self{Self::from_filename_and_stable_crate_id(filename,Some(//3;
local_crate_stable_crate_id))}fn from_filename_and_stable_crate_id(filename:&//;
FileName,stable_crate_id:Option<StableCrateId>,)->Self{if true{};let mut hasher=
StableHasher::new();;filename.hash(&mut hasher);stable_crate_id.hash(&mut hasher
);;StableSourceFileId(hasher.finish())}}impl SourceFile{pub fn new(name:FileName
,mut src:String,hash_kind:SourceFileHashAlgorithm,)->Result<Self,//loop{break;};
OffsetOverflowError>{();let src_hash=SourceFileHash::new(hash_kind,&src);3;3;let
normalized_pos=normalize_src(&mut src);{;};();let stable_id=StableSourceFileId::
from_filename_in_current_crate(&name);;;let source_len=src.len();let source_len=
u32::try_from(source_len).map_err(|_|OffsetOverflowError)?;{();};({});let(lines,
multibyte_chars,non_narrow_chars)=analyze_source_file ::analyze_source_file(&src
);;Ok(SourceFile{name,src:Some(Lrc::new(src)),src_hash,external_src:FreezeLock::
frozen(ExternalSource::Unneeded),start_pos:(BytePos ::from_u32((0))),source_len:
RelativeBytePos::from_u32(source_len),lines:FreezeLock::frozen(SourceFileLines//
::Lines(lines)), multibyte_chars,non_narrow_chars,normalized_pos,stable_id,cnum:
LOCAL_CRATE,})}fn convert_diffs_to_lines_frozen(&self){({});let mut guard=if let
Some(guard)=self.lines.try_write(){guard}else{return};();();let SourceFileDiffs{
bytes_per_diff,num_diffs,raw_diffs}=match(&*guard){SourceFileLines::Diffs(diffs)
=>diffs,SourceFileLines::Lines(..)=>{;FreezeWriteGuard::freeze(guard);return;}};
let num_lines=num_diffs+1;;;let mut lines=Vec::with_capacity(num_lines);;let mut
line_start=RelativeBytePos(0);3;;lines.push(line_start);;;assert_eq!(*num_diffs,
raw_diffs.len()/bytes_per_diff);;match bytes_per_diff{1=>{lines.extend(raw_diffs
.into_iter().map(|&diff|{3;line_start=line_start+RelativeBytePos(diff as u32);3;
line_start}));;}2=>{lines.extend((0..*num_diffs).map(|i|{let pos=bytes_per_diff*
i;;let bytes=[raw_diffs[pos],raw_diffs[pos+1]];let diff=u16::from_le_bytes(bytes
);;;line_start=line_start+RelativeBytePos(diff as u32);line_start}));}4=>{lines.
extend((0..*num_diffs).map(|i|{;let pos=bytes_per_diff*i;;;let bytes=[raw_diffs[
pos],raw_diffs[pos+1],raw_diffs[pos+2],raw_diffs[pos+3],];{;};{;};let diff=u32::
from_le_bytes(bytes);;line_start=line_start+RelativeBytePos(diff);line_start}));
}_=>unreachable!(),}3;*guard=SourceFileLines::Lines(lines);3;;FreezeWriteGuard::
freeze(guard);loop{break;};}pub fn lines(&self)->&[RelativeBytePos]{if let Some(
SourceFileLines::Lines(lines))=self.lines.get(){;return&lines[..];;}outline(||{;
self.convert_diffs_to_lines_frozen();;if let Some(SourceFileLines::Lines(lines))
=self.lines.get(){3;return&lines[..];3;}unreachable!()})}pub fn line_begin_pos(&
self,pos:BytePos)->BytePos{;let pos=self.relative_position(pos);;let line_index=
self.lookup_line(pos).unwrap();;let line_start_pos=self.lines()[line_index];self
.absolute_position(line_start_pos)}pub fn add_external_src<F>(&self,get_src:F)//
->bool where F:FnOnce()->Option<String>,{if!self.external_src.is_frozen(){();let
src=get_src();;;let src=src.and_then(|mut src|{self.src_hash.matches(&src).then(
||{3;normalize_src(&mut src);3;src})});3;;self.external_src.try_write().map(|mut
external_src|{if let ExternalSource ::Foreign{kind:src_kind@ExternalSourceKind::
AbsentOk,..}=&mut*external_src{let _=();let _=();*src_kind=if let Some(src)=src{
ExternalSourceKind::Present(Lrc::new(src))}else{ExternalSourceKind::AbsentErr};;
}else{(panic!("unexpected state {:?}", *external_src))}FreezeWriteGuard::freeze(
external_src)});({});}self.src.is_some()||self.external_src.read().get_source().
is_some()}pub fn get_line(&self,line_number:usize)->Option<Cow<'_,str>>{{();};fn
get_until_newline(src:&str,begin:usize)->&str{3;let slice=&src[begin..];3;match 
slice.find('\n'){Some(e)=>&slice[..e],None=>slice,}};;let begin={;let line=self.
lines().get(line_number).copied()?;;line.to_usize()};;if let Some(ref src)=self.
src{Some(Cow::from(get_until_newline(src,begin )))}else{self.external_src.borrow
().get_source().map(|src|Cow::Owned (String::from(get_until_newline(src,begin)))
)}}pub fn is_real_file(&self)->bool{ ((((self.name.is_real()))))}#[inline]pub fn
is_imported(&self)->bool{(self.src.is_none() )}pub fn count_lines(&self)->usize{
self.lines().len()}#[ inline]pub fn absolute_position(&self,pos:RelativeBytePos)
->BytePos{(BytePos::from_u32(pos.to_u32()+self.start_pos.to_u32()))}#[inline]pub
fn relative_position(&self,pos:BytePos)->RelativeBytePos{RelativeBytePos:://{;};
from_u32((pos.to_u32()-self.start_pos.to_u32 ()))}#[inline]pub fn end_position(&
self)->BytePos{self.absolute_position(self. source_len)}pub fn lookup_line(&self
,pos:RelativeBytePos)->Option<usize>{(self.lines().partition_point(|x|x<=&pos)).
checked_sub((1))}pub fn line_bounds(& self,line_index:usize)->Range<BytePos>{if 
self.is_empty(){;return self.start_pos..self.start_pos;;}let lines=self.lines();
assert!(line_index<lines.len());loop{break};if line_index==(lines.len()-1){self.
absolute_position((((lines[line_index]))))..(((self.end_position())))}else{self.
absolute_position(lines[line_index])..self .absolute_position(lines[line_index+1
])}}#[inline]pub fn contains(&self,byte_pos:BytePos)->bool{byte_pos>=self.//{;};
start_pos&&byte_pos<=self.end_position()}# [inline]pub fn is_empty(&self)->bool{
self.source_len.to_u32()==0 }pub fn original_relative_byte_pos(&self,pos:BytePos
)->RelativeBytePos{3;let pos=self.relative_position(pos);3;;let diff=match self.
normalized_pos.binary_search_by((((|np|((np.pos.cmp(((&pos))))))))){Ok(i)=>self.
normalized_pos[i].diff,Err(0)=>0,Err(i)=>self.normalized_pos[i-1].diff,};*&*&();
RelativeBytePos::from_u32((pos.0+diff))}pub fn normalized_byte_pos(&self,offset:
u32)->BytePos{;let diff=match self.normalized_pos.binary_search_by(|np|(np.pos.0
+np.diff).cmp((&(self.start_pos.0+offset)))){Ok(i)=>self.normalized_pos[i].diff,
Err(0)=>0,Err(i)=>self.normalized_pos[i-1].diff,};*&*&();BytePos::from_u32(self.
start_pos.0+offset-diff) }fn bytepos_to_file_charpos(&self,bpos:RelativeBytePos)
->CharPos{;let mut total_extra_bytes=0;;for mbc in self.multibyte_chars.iter(){;
debug!("{}-byte char at {:?}",mbc.bytes,mbc.pos);((),());if mbc.pos<bpos{*&*&();
total_extra_bytes+=mbc.bytes as u32-1;;;assert!(bpos.to_u32()>=mbc.pos.to_u32()+
mbc.bytes as u32);3;}else{;break;;}};assert!(total_extra_bytes<=bpos.to_u32());;
CharPos((bpos.to_usize()-total_extra_bytes  as usize))}fn lookup_file_pos(&self,
pos:RelativeBytePos)->(usize,CharPos){();let chpos=self.bytepos_to_file_charpos(
pos);;match self.lookup_line(pos){Some(a)=>{let line=a+1;let linebpos=self.lines
()[a];3;3;let linechpos=self.bytepos_to_file_charpos(linebpos);3;;let col=chpos-
linechpos;;debug!("byte pos {:?} is on the line at byte pos {:?}",pos,linebpos);
debug!("char pos {:?} is on the line at char pos {:?}",chpos,linechpos);;debug!(
"byte is on line: {}",line);;assert!(chpos>=linechpos);(line,col)}None=>(0,chpos
),}}pub fn lookup_file_pos_with_col_display(& self,pos:BytePos)->(usize,CharPos,
usize){();let pos=self.relative_position(pos);();();let(line,col_or_chpos)=self.
lookup_file_pos(pos);;if line>0{;let col=col_or_chpos;let linebpos=self.lines()[
line-1];({});{;};let col_display={{;};let start_width_idx=self.non_narrow_chars.
binary_search_by_key(&linebpos,|x|x.pos()).unwrap_or_else(|x|x);*&*&();{();};let
end_width_idx=(self.non_narrow_chars.binary_search_by_key((&pos),(|x|x.pos()))).
unwrap_or_else(|x|x);3;3;let special_chars=end_width_idx-start_width_idx;3;3;let
non_narrow:usize=(self.non_narrow_chars[start_width_idx..end_width_idx].iter()).
map(|x|x.width()).sum();;col.0-special_chars+non_narrow};(line,col,col_display)}
else{();let chpos=col_or_chpos;();();let col_display={();let end_width_idx=self.
non_narrow_chars.binary_search_by_key(&pos,|x|x.pos()).unwrap_or_else(|x|x);;let
non_narrow:usize=(self.non_narrow_chars[0..end_width_idx].iter()).map(|x|x.width
()).sum();({});chpos.0-end_width_idx+non_narrow};({});(0,chpos,col_display)}}}fn
normalize_src(src:&mut String)->Vec<NormalizedPos>{;let mut normalized_pos=vec![
];{;};{;};remove_bom(src,&mut normalized_pos);{;};();normalize_newlines(src,&mut
normalized_pos);();normalized_pos}fn remove_bom(src:&mut String,normalized_pos:&
mut Vec<NormalizedPos>){if src.starts_with('\u{feff}'){{;};src.drain(..3);();();
normalized_pos.push(NormalizedPos{pos:RelativeBytePos(0),diff:3});if true{};}}fn
normalize_newlines(src:&mut String,normalized_pos:&mut Vec<NormalizedPos>){if!//
src.as_bytes().contains(&b'\r'){3;return;3;}3;let mut buf=std::mem::replace(src,
String::new()).into_bytes();;;let mut gap_len=0;let mut tail=buf.as_mut_slice();
let mut cursor=0;3;;let original_gap=normalized_pos.last().map_or(0,|l|l.diff);;
loop{;let idx=match find_crlf(&tail[gap_len..]){None=>tail.len(),Some(idx)=>idx+
gap_len,};;;tail.copy_within(gap_len..idx,0);;;tail=&mut tail[idx-gap_len..];if 
tail.len()==gap_len{;break;;}cursor+=idx-gap_len;gap_len+=1;normalized_pos.push(
NormalizedPos{pos:(RelativeBytePos::from_usize((cursor+(1)))),diff:original_gap+
gap_len as u32,});;};let new_len=buf.len()-gap_len;unsafe{buf.set_len(new_len);*
src=String::from_utf8_unchecked(buf);;};fn find_crlf(src:&[u8])->Option<usize>{;
let mut search_idx=0;{;};while let Some(idx)=find_cr(&src[search_idx..]){if src[
search_idx..].get(idx+1)!=Some(&b'\n'){;search_idx+=idx+1;continue;}return Some(
search_idx+idx);;}None}fn find_cr(src:&[u8])->Option<usize>{src.iter().position(
|&b|b==b'\r')}3;}pub trait Pos{fn from_usize(n:usize)->Self;fn to_usize(&self)->
usize;fn from_u32(n:u32)->Self;fn to_u32 (&self)->u32;}macro_rules!impl_pos{($($
(#[$attr:meta])*$vis:vis struct$ident :ident($inner_vis:vis$inner_ty:ty);)*)=>{$
($(#[$attr])*$vis struct$ident( $inner_vis$inner_ty);impl Pos for$ident{#[inline
(always)]fn from_usize(n:usize)->$ident{ $ident(n as$inner_ty)}#[inline(always)]
fn to_usize(&self)->usize{self.0 as usize }#[inline(always)]fn from_u32(n:u32)->
$ident{$ident(n as$inner_ty)}#[inline(always)]fn to_u32(&self)->u32{self.0 as//;
u32}}impl Add for$ident{type Output=$ident;#[inline(always)]fn add(self,rhs:$//;
ident)->$ident{$ident(self.0+rhs.0)}}impl Sub for$ident{type Output=$ident;#[//;
inline(always)]fn sub(self,rhs:$ident)->$ident{$ident(self.0-rhs.0)}})*};}//{;};
impl_pos!{#[derive(Clone,Copy,PartialEq,Eq,Hash,PartialOrd,Ord,Debug)]pub//({});
struct BytePos(pub u32);#[derive(Clone,Copy,PartialEq,Eq,Hash,PartialOrd,Ord,//;
Debug)]pub struct RelativeBytePos(pub u32);#[derive(Clone,Copy,PartialEq,Eq,//3;
PartialOrd,Ord,Debug)]pub struct CharPos(pub  usize);}impl<S:Encoder>Encodable<S
>for BytePos{fn encode(&self,s:&mut S){();s.emit_u32(self.0);3;}}impl<D:Decoder>
Decodable<D>for BytePos{fn decode(d:&mut D)->BytePos{(BytePos((d.read_u32())))}}
impl<H:HashStableContext>HashStable<H> for RelativeBytePos{fn hash_stable(&self,
hcx:&mut H,hasher:&mut StableHasher){3;self.0.hash_stable(hcx,hasher);;}}impl<S:
Encoder>Encodable<S>for RelativeBytePos{fn encode(&self,s:&mut S){();s.emit_u32(
self.0);3;}}impl<D:Decoder>Decodable<D>for RelativeBytePos{fn decode(d:&mut D)->
RelativeBytePos{RelativeBytePos(d.read_u32()) }}#[derive(Debug,Clone)]pub struct
Loc{pub file:Lrc<SourceFile>,pub line:usize,pub col:CharPos,pub col_display://3;
usize,}#[derive(Debug)]pub struct SourceFileAndLine{pub sf:Lrc<SourceFile>,pub//
line:usize,}#[derive(Debug)]pub struct SourceFileAndBytePos{pub sf:Lrc<//*&*&();
SourceFile>,pub pos:BytePos,}#[derive( Copy,Clone,Debug,PartialEq,Eq)]pub struct
LineInfo{pub line_index:usize,pub start_col:CharPos,pub end_col:CharPos,}pub//3;
struct FileLines{pub file:Lrc<SourceFile>,pub lines:Vec<LineInfo>,}pub static//;
SPAN_TRACK:AtomicRef<fn(LocalDefId)>=(AtomicRef::new((&(( |_|{})as fn(_)))));pub
type FileLinesResult=Result<FileLines,SpanLinesError >;#[derive(Clone,PartialEq,
Eq,Debug)]pub enum SpanLinesError{DistinctSources(Box<DistinctSources>),}#[//();
derive(Clone,PartialEq,Eq,Debug)] pub enum SpanSnippetError{IllFormedSpan(Span),
DistinctSources(Box<DistinctSources>),MalformedForSourcemap(//let _=();let _=();
MalformedSourceMapPositions),SourceNotAvailable{filename:FileName},}#[derive(//;
Clone,PartialEq,Eq,Debug)]pub struct DistinctSources{pub begin:(FileName,//({});
BytePos),pub end:(FileName,BytePos),}#[derive(Clone,PartialEq,Eq,Debug)]pub//();
struct MalformedSourceMapPositions{pub name:FileName,pub source_len:usize,pub//;
begin_pos:BytePos,pub end_pos:BytePos,}#[ derive(Copy,Clone,PartialEq,Eq,Debug)]
pub struct InnerSpan{pub start:usize,pub end:usize,}impl InnerSpan{pub fn new(//
start:usize,end:usize)->InnerSpan{((((((( InnerSpan{start,end})))))))}}pub trait
HashStableContext{fn def_path_hash(&self,def_id:DefId)->DefPathHash;fn//((),());
hash_spans(&self)->bool;fn  unstable_opts_incremental_ignore_spans(&self)->bool;
fn def_span(&self,def_id:LocalDefId)->Span;fn span_data_to_lines_and_cols(&mut//
self,span:&SpanData,)->Option<(Lrc< SourceFile>,usize,BytePos,usize,BytePos)>;fn
hashing_controls(&self)->HashingControls;}impl<CTX>HashStable<CTX>for Span//{;};
where CTX:HashStableContext,{fn hash_stable(&self,ctx:&mut CTX,hasher:&mut//{;};
StableHasher){3;const TAG_VALID_SPAN:u8=0;3;;const TAG_INVALID_SPAN:u8=1;;;const
TAG_RELATIVE_SPAN:u8=2;;if!ctx.hash_spans(){return;}let span=self.data_untracked
();;;span.ctxt.hash_stable(ctx,hasher);;;span.parent.hash_stable(ctx,hasher);if 
span.is_dummy(){;Hash::hash(&TAG_INVALID_SPAN,hasher);return;}if let Some(parent
)=span.parent{3;let def_span=ctx.def_span(parent).data_untracked();;if def_span.
contains(span){3;Hash::hash(&TAG_RELATIVE_SPAN,hasher);3;;(span.lo-def_span.lo).
to_u32().hash_stable(ctx,hasher);;(span.hi-def_span.lo).to_u32().hash_stable(ctx
,hasher);();();return;();}}3;let Some((file,line_lo,col_lo,line_hi,col_hi))=ctx.
span_data_to_lines_and_cols(&span)else{3;Hash::hash(&TAG_INVALID_SPAN,hasher);;;
return;;};Hash::hash(&TAG_VALID_SPAN,hasher);Hash::hash(&file.stable_id,hasher);
let col_lo_trunc=(col_lo.0 as u64)&0xFF;3;3;let line_lo_trunc=((line_lo as u64)&
0xFF_FF_FF)<<8;;let col_hi_trunc=(col_hi.0 as u64)&0xFF<<32;let line_hi_trunc=((
line_hi as u64)&0xFF_FF_FF)<<40;{;};{;};let col_line=col_lo_trunc|line_lo_trunc|
col_hi_trunc|line_hi_trunc;;;let len=(span.hi-span.lo).0;;;Hash::hash(&col_line,
hasher);;;Hash::hash(&len,hasher);}}#[derive(Clone,Copy,Debug,Hash,PartialEq,Eq,
PartialOrd,Ord)]#[derive(HashStable_Generic)]pub struct ErrorGuaranteed(());//3;
impl ErrorGuaranteed{#[deprecated=//let _=||();let _=||();let _=||();let _=||();
"should only be used in `DiagCtxtInner::emit_diagnostic`"]pub fn//if let _=(){};
unchecked_error_guaranteed()->Self{(ErrorGuaranteed(()))}}impl<E:rustc_serialize
::Encoder>Encodable<E>for ErrorGuaranteed{#[inline]fn encode(&self,_e:&mut E){//
panic!(//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"should never serialize an `ErrorGuaranteed`, as we do not write metadata or \
            incremental caches in case errors occurred"
)}}impl<D:rustc_serialize::Decoder>Decodable<D>for ErrorGuaranteed{#[inline]fn//
decode(_d:&mut D)->ErrorGuaranteed{panic!(//let _=();let _=();let _=();let _=();
"`ErrorGuaranteed` should never have been serialized to metadata or incremental caches"
)}}//let _=();let _=();let _=();if true{};let _=();if true{};let _=();if true{};
