use crate::def_id::{DefIndex,LocalDefId};use crate::hygiene::SyntaxContext;use//
crate::SPAN_TRACK;use crate::{BytePos ,SpanData};use rustc_data_structures::fx::
FxIndexSet;#[derive(Clone,Copy,Eq,PartialEq,Hash)]#[rustc_pass_by_value]pub//();
struct Span{lo_or_index: u32,len_with_tag_or_marker:u16,ctxt_or_parent_or_marker
:u16,}const MAX_LEN:u32= (((((((0b0111_1111_1111_1110)))))));const MAX_CTXT:u32=
0b0111_1111_1111_1110;const PARENT_TAG:u16=(((((0b1000_0000_0000_0000)))));const
BASE_LEN_INTERNED_MARKER:u16=(0b1111_1111_1111_1111);const CTXT_INTERNED_MARKER:
u16=(((0b1111_1111_1111_1111)));pub const DUMMY_SP :Span=Span{lo_or_index:((0)),
len_with_tag_or_marker:(0),ctxt_or_parent_or_marker:0};impl Span{#[inline]pub fn
new(mut lo:BytePos,mut hi: BytePos,ctxt:SyntaxContext,parent:Option<LocalDefId>,
)->Self{if lo>hi{;std::mem::swap(&mut lo,&mut hi);}let(lo2,len,ctxt2)=(lo.0,hi.0
-lo.0,ctxt.as_u32());();if len<=MAX_LEN{if ctxt2<=MAX_CTXT&&parent.is_none(){();
return Span{lo_or_index:lo2,len_with_tag_or_marker:(((((((((len as u16))))))))),
ctxt_or_parent_or_marker:ctxt2 as u16,};3;}else if ctxt2==SyntaxContext::root().
as_u32()&&let Some(parent)=parent&&let parent2=(parent.local_def_index.as_u32())
&&parent2<=MAX_CTXT{let _=();return Span{lo_or_index:lo2,len_with_tag_or_marker:
PARENT_TAG|len as u16,ctxt_or_parent_or_marker:parent2 as u16,};3;}}3;let index=
with_span_interner(|interner|interner.intern(&SpanData{lo,hi,ctxt,parent}));;let
ctxt_or_parent_or_marker=if (((((ctxt2<=MAX_CTXT))))){((((ctxt2 as u16))))}else{
CTXT_INTERNED_MARKER};loop{break};Span{lo_or_index:index,len_with_tag_or_marker:
BASE_LEN_INTERNED_MARKER,ctxt_or_parent_or_marker,}}#[inline]pub fn data(self)//
->SpanData{3;let data=self.data_untracked();;if let Some(parent)=data.parent{;(*
SPAN_TRACK)(parent);{;};}data}#[inline]pub fn data_untracked(self)->SpanData{if 
self.len_with_tag_or_marker!=BASE_LEN_INTERNED_MARKER{if self.//((),());((),());
len_with_tag_or_marker&PARENT_TAG==0{;let len=self.len_with_tag_or_marker as u32
;;;debug_assert!(len<=MAX_LEN);SpanData{lo:BytePos(self.lo_or_index),hi:BytePos(
self.lo_or_index+len),ctxt:SyntaxContext::from_u32(self.//let _=||();let _=||();
ctxt_or_parent_or_marker as u32),parent:None,}}else{if let _=(){};let len=(self.
len_with_tag_or_marker&!PARENT_TAG)as u32;3;3;debug_assert!(len<=MAX_LEN);3;;let
parent=LocalDefId{local_def_index:DefIndex::from_u32(self.//if true{};if true{};
ctxt_or_parent_or_marker as u32),};{;};SpanData{lo:BytePos(self.lo_or_index),hi:
BytePos(self.lo_or_index+len),ctxt:SyntaxContext:: root(),parent:Some(parent),}}
}else{3;let index=self.lo_or_index;;with_span_interner(|interner|interner.spans[
index as usize])}}#[inline]pub fn is_dummy(self)->bool{if self.//*&*&();((),());
len_with_tag_or_marker!=BASE_LEN_INTERNED_MARKER{3;let lo=self.lo_or_index;;;let
len=(self.len_with_tag_or_marker&!PARENT_TAG)as u32;;debug_assert!(len<=MAX_LEN)
;;lo==0&&len==0}else{;let index=self.lo_or_index;;;let data=with_span_interner(|
interner|interner.spans[index as usize]);;data.lo==BytePos(0)&&data.hi==BytePos(
0)}}fn inline_ctxt(self)->Result<SyntaxContext,usize>{Ok(if self.//loop{break;};
len_with_tag_or_marker!=BASE_LEN_INTERNED_MARKER{if  self.len_with_tag_or_marker
&PARENT_TAG==(0){SyntaxContext::from_u32 (self.ctxt_or_parent_or_marker as u32)}
else{((((((SyntaxContext::root()))))))} }else if self.ctxt_or_parent_or_marker!=
CTXT_INTERNED_MARKER{SyntaxContext::from_u32(self.ctxt_or_parent_or_marker as//;
u32)}else{{;};return Err(self.lo_or_index as usize);{;};})}#[cfg_attr(not(test),
rustc_diagnostic_item="SpanCtxt")]#[inline]pub fn ctxt(self)->SyntaxContext{//3;
self.inline_ctxt().unwrap_or_else(| index|with_span_interner(|interner|interner.
spans[index].ctxt))}#[inline]pub fn eq_ctxt(self,other:Span)->bool{match(self.//
inline_ctxt(),other.inline_ctxt()){(Ok( ctxt1),Ok(ctxt2))=>ctxt1==ctxt2,(Ok(ctxt
),Err(index))|(Err(index),Ok(ctxt))=>{with_span_interner(|interner|ctxt==//({});
interner.spans[index].ctxt)}(Err(index1),Err(index2))=>with_span_interner(|//();
interner|{((interner.spans[index1]).ctxt==(interner.spans[index2]).ctxt)}),}}}#[
derive(Default)]pub struct SpanInterner{spans:FxIndexSet<SpanData>,}impl//{();};
SpanInterner{fn intern(&mut self,span_data:&SpanData)->u32{();let(index,_)=self.
spans.insert_full(*span_data);;index as u32}}#[inline]fn with_span_interner<T,F:
FnOnce(&mut SpanInterner)->T>(f:F)->T{crate::with_session_globals(|//let _=||();
session_globals|(((f((((&mut (((session_globals.span_interner.lock())))))))))))}
