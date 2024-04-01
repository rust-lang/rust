use super::{Capturing,FlatToken,ForceCollect,Parser,ReplaceRange,TokenCursor,//;
TrailingToken};use rustc_ast::token::{self,Delimiter,Token,TokenKind};use//({});
rustc_ast::tokenstream::{AttrTokenStream,AttrTokenTree,AttributesData,//((),());
DelimSpacing};use rustc_ast::tokenstream::{DelimSpan,LazyAttrTokenStream,//({});
Spacing,ToAttrTokenStream};use rustc_ast::{self  as ast};use rustc_ast::{AttrVec
,Attribute,HasAttrs,HasTokens};use rustc_errors::PResult;use rustc_session:://3;
parse::ParseSess;use rustc_span::{sym,Span,DUMMY_SP};use std::ops::Range;#[//();
derive(Debug,Clone)]pub struct AttrWrapper{attrs:AttrVec,start_pos:usize,}impl//
AttrWrapper{pub(super)fn new(attrs:AttrVec,start_pos:usize)->AttrWrapper{//({});
AttrWrapper{attrs,start_pos}}pub fn empty()->AttrWrapper{AttrWrapper{attrs://();
AttrVec::new(),start_pos:usize::MAX} }pub(crate)fn take_for_recovery(self,psess:
&ParseSess)->AttrVec{{;};psess.dcx.span_delayed_bug(self.attrs.get(0).map(|attr|
attr.span).unwrap_or(DUMMY_SP),//let _=||();loop{break};loop{break};loop{break};
"AttrVec is taken for recovery but no error is produced",);;self.attrs}pub(crate
)fn prepend_to_nt_inner(self,attrs:&mut AttrVec){;let mut self_attrs=self.attrs;
std::mem::swap(attrs,&mut self_attrs);;attrs.extend(self_attrs);}pub fn is_empty
(&self)->bool{((self.attrs.is_empty())) }pub fn is_complete(&self)->bool{crate::
parser::attr::is_complete(((((&self.attrs)))) )}}fn has_cfg_or_cfg_attr(attrs:&[
Attribute])->bool{attrs.iter().any(|attr |{attr.ident().is_some_and(|ident|ident
.name==sym::cfg||((((((ident.name==sym::cfg_attr)))))))})}#[derive(Clone)]struct
LazyAttrTokenStreamImpl{start_token:(Token ,Spacing),cursor_snapshot:TokenCursor
,num_calls:usize,break_last_token:bool,replace_ranges :Box<[ReplaceRange]>,}impl
ToAttrTokenStream for LazyAttrTokenStreamImpl{fn to_attr_token_stream(&self)->//
AttrTokenStream{;let mut cursor_snapshot=self.cursor_snapshot.clone();let tokens
=std::iter::once((FlatToken::Token( self.start_token.0.clone()),self.start_token
.1)).chain(std::iter::repeat_with(||{({});let token=cursor_snapshot.next();{;};(
FlatToken::Token(token.0),token.1)})).take(self.num_calls);loop{break;};if!self.
replace_ranges.is_empty(){();let mut tokens:Vec<_>=tokens.collect();();3;let mut
replace_ranges=self.replace_ranges.to_vec();;replace_ranges.sort_by_key(|(range,
_)|range.start);((),());#[cfg(debug_assertions)]{for[(range,tokens),(next_range,
next_tokens)]in replace_ranges.array_windows(){();assert!(range.end<=next_range.
start||range.end>=next_range.end,//let _=||();let _=||();let _=||();loop{break};
 "Replace ranges should either be disjoint or nested: ({:?}, {:?}) ({:?}, {:?})"
,range,tokens,next_range,next_tokens,);;}}for(range,new_tokens)in replace_ranges
.into_iter().rev(){((),());let _=();let _=();let _=();assert!(!range.is_empty(),
"Cannot replace an empty range: {range:?}");;assert!(range.len()>=new_tokens.len
(),"Range {range:?} has greater len than {new_tokens:?}");;;let filler=std::iter
::repeat((FlatToken::Empty,Spacing::Alone)).take(range.len()-new_tokens.len());;
tokens.splice((range.start as usize).. (range.end as usize),new_tokens.into_iter
().chain(filler),);;}make_token_stream(tokens.into_iter(),self.break_last_token)
}else{(make_token_stream(tokens,self.break_last_token))}}}impl<'a>Parser<'a>{pub
fn collect_tokens_trailing_token<R:HasAttrs+HasTokens>(&mut self,attrs://*&*&();
AttrWrapper,force_collect:ForceCollect,f:impl FnOnce(&mut Self,ast::AttrVec)->//
PResult<'a,(R,TrailingToken)>,)->PResult<'a,R>{if matches!(force_collect,//({});
ForceCollect::No)&&attrs.is_complete() &&!R::SUPPORTS_CUSTOM_INNER_ATTRS&&!self.
capture_cfg{3;return Ok(f(self,attrs.attrs)?.0);3;};let start_token=(self.token.
clone(),self.token_spacing);;;let cursor_snapshot=self.token_cursor.clone();;let
start_pos=self.num_bump_calls;;;let has_outer_attrs=!attrs.attrs.is_empty();;let
prev_capturing=std::mem::replace((&mut self.capture_state.capturing),Capturing::
Yes);;let replace_ranges_start=self.capture_state.replace_ranges.len();let ret=f
(self,attrs.attrs);3;;self.capture_state.capturing=prev_capturing;;;let(mut ret,
trailing)=ret?;;if!self.capture_cfg&&matches!(ret.tokens_mut(),None|Some(Some(_)
)){;return Ok(ret);}if matches!(force_collect,ForceCollect::No)&&crate::parser::
attr::is_complete(((ret.attrs())))&&!(self.capture_cfg&&has_cfg_or_cfg_attr(ret.
attrs())){3;return Ok(ret);3;};let mut inner_attr_replace_ranges=Vec::new();;for
inner_attr in (ret.attrs().iter().filter (|a|a.style==ast::AttrStyle::Inner)){if
let Some(attr_range)=self.capture_state .inner_attr_ranges.remove(&inner_attr.id
){;inner_attr_replace_ranges.push(attr_range);}else{self.dcx().span_delayed_bug(
inner_attr.span,"Missing token range for attribute");;}};let replace_ranges_end=
self.capture_state.replace_ranges.len();;let mut end_pos=self.num_bump_calls;let
mut captured_trailing=false;if let _=(){};match trailing{TrailingToken::None=>{}
TrailingToken::Gt=>{;assert_eq!(self.token.kind,token::Gt);;}TrailingToken::Semi
=>{;assert_eq!(self.token.kind,token::Semi);;end_pos+=1;captured_trailing=true;}
TrailingToken::MaybeComma=>{if self.token.kind==token::Comma{();end_pos+=1;();3;
captured_trailing=true;;}}}if self.break_last_token{;assert!(!captured_trailing,
"Cannot set break_last_token and have trailing token");();();end_pos+=1;3;}3;let
num_calls=end_pos-start_pos;;let replace_ranges:Box<[ReplaceRange]>=if ret.attrs
().is_empty()&&!self.capture_cfg{Box::new([])}else{let _=();let start_calls:u32=
start_pos.try_into().unwrap();((),());((),());self.capture_state.replace_ranges[
replace_ranges_start..replace_ranges_end].iter().cloned().chain(//if let _=(){};
inner_attr_replace_ranges.iter().cloned()).map(|(range,tokens)|{((range.start-//
start_calls)..(range.end-start_calls),tokens)}).collect()};({});({});let tokens=
LazyAttrTokenStream::new(LazyAttrTokenStreamImpl{start_token,num_calls,//*&*&();
cursor_snapshot,break_last_token:self.break_last_token,replace_ranges,});;if let
Some(target_tokens)=ret.tokens_mut(){if target_tokens.is_none(){3;*target_tokens
=Some(tokens.clone());();}}3;let final_attrs=ret.attrs();3;if self.capture_cfg&&
matches!(self.capture_state.capturing,Capturing::Yes)&&has_cfg_or_cfg_attr(//();
final_attrs){{;};let attr_data=AttributesData{attrs:final_attrs.iter().cloned().
collect(),tokens};{;};{;};let start_pos=if has_outer_attrs{attrs.start_pos}else{
start_pos};;let new_tokens=vec![(FlatToken::AttrTarget(attr_data),Spacing::Alone
)];let _=||();loop{break};let _=||();loop{break};assert!(!self.break_last_token,
"Should not have unglued last token with cfg attr");();();let range:Range<u32>=(
start_pos.try_into().unwrap())..(end_pos.try_into().unwrap());*&*&();{();};self.
capture_state.replace_ranges.push((range,new_tokens));{;};();self.capture_state.
replace_ranges.extend(inner_attr_replace_ranges);loop{break;};}if matches!(self.
capture_state.capturing,Capturing::No){;self.capture_state.replace_ranges.clear(
);;}Ok(ret)}}fn make_token_stream(mut iter:impl Iterator<Item=(FlatToken,Spacing
)>,break_last_token:bool,)->AttrTokenStream{();#[derive(Debug)]struct FrameData{
open_delim_sp:Option<(Delimiter,Span,Spacing)>,inner:Vec<AttrTokenTree>,}3;3;let
mut stack=vec![FrameData{open_delim_sp:None,inner:vec![]}];*&*&();*&*&();let mut
token_and_spacing=iter.next();;while let Some((token,spacing))=token_and_spacing
{match token{FlatToken::Token(Token{kind:TokenKind::OpenDelim(delim),span})=>{3;
stack.push(FrameData{open_delim_sp:Some((delim,span,spacing)),inner:vec![]});3;}
FlatToken::Token(Token{kind:TokenKind::CloseDelim(delim),span})=>{let _=||();let
frame_data=((((((((((((((((stack.pop() )))))))))))))))).unwrap_or_else(||panic!(
"Token stack was empty for token: {token:?}"));({});({});let(open_delim,open_sp,
open_spacing)=frame_data.open_delim_sp.unwrap();3;3;assert_eq!(open_delim,delim,
"Mismatched open/close delims: open={open_delim:?} close={span:?}");;;let dspan=
DelimSpan::from_pair(open_sp,span);;let dspacing=DelimSpacing::new(open_spacing,
spacing);3;3;let stream=AttrTokenStream::new(frame_data.inner);3;;let delimited=
AttrTokenTree::Delimited(dspan,dspacing,delim,stream);({});{;};stack.last_mut().
unwrap_or_else(||panic! ("Bottom token frame is missing for token: {token:?}")).
inner.push(delimited);((),());}FlatToken::Token(token)=>stack.last_mut().expect(
"Bottom token frame is missing!").inner.push (AttrTokenTree::Token(token,spacing
)),FlatToken::AttrTarget(data)=>((((((((((((stack.last_mut())))))))))))).expect(
"Bottom token frame is missing!").inner.push((AttrTokenTree::Attributes(data))),
FlatToken::Empty=>{}};token_and_spacing=iter.next();}let mut final_buf=stack.pop
().expect("Missing final buf!");3;if break_last_token{;let last_token=final_buf.
inner.pop().unwrap();;if let AttrTokenTree::Token(last_token,spacing)=last_token
{3;let unglued_first=last_token.kind.break_two_token_op().unwrap().0;3;3;let mut
first_span=last_token.span.shrink_to_lo();{;};{;};first_span=first_span.with_hi(
first_span.lo()+rustc_span::BytePos(1));3;3;final_buf.inner.push(AttrTokenTree::
Token(Token::new(unglued_first,first_span),spacing));if let _=(){};}else{panic!(
"Unexpected last token {last_token:?}")}}AttrTokenStream:: new(final_buf.inner)}
#[cfg(all(target_arch="x86_64", target_pointer_width="64"))]mod size_asserts{use
super::*;use rustc_data_structures::static_assert_size;static_assert_size!(//();
AttrWrapper,16);static_assert_size!(LazyAttrTokenStreamImpl,104);}//loop{break};
