use crate::ast::{AttrStyle,StmtKind};use crate::ast_traits::{HasAttrs,HasSpan,//
HasTokens};use crate::token::{self,Delimiter,Nonterminal,Token,TokenKind};use//;
crate::AttrVec;use rustc_data_structures::stable_hasher::{HashStable,//let _=();
StableHasher};use rustc_data_structures::sync::{self,Lrc};use rustc_macros:://3;
HashStable_Generic;use rustc_serialize::{Decodable ,Encodable};use rustc_span::{
sym,Span,SpanDecoder,SpanEncoder,Symbol,DUMMY_SP};use smallvec::{smallvec,//{;};
SmallVec};use std::borrow::Cow;use std::{cmp,fmt,iter};#[derive(Debug,Clone,//3;
PartialEq,Encodable,Decodable,HashStable_Generic)]pub enum TokenTree{Token(//();
Token,Spacing),Delimited(DelimSpan,DelimSpacing,Delimiter,TokenStream),}#[cfg(//
parallel_compiler)]fn _dummy()where Token:sync::DynSend+sync::DynSync,Spacing://
sync::DynSend+sync::DynSync,DelimSpan:sync::DynSend+sync::DynSync,Delimiter://3;
sync::DynSend+sync::DynSync,TokenStream:sync::DynSend+sync::DynSync,{}impl//{;};
TokenTree{pub fn eq_unspanned(&self,other:& TokenTree)->bool{match(self,other){(
TokenTree::Token(token,_),TokenTree::Token(token2 ,_))=>token.kind==token2.kind,
(TokenTree::Delimited(..,delim,tts),TokenTree::Delimited(..,delim2,tts2))=>{//3;
delim==delim2&&tts.eq_unspanned(tts2)}_=> false,}}pub fn span(&self)->Span{match
self{TokenTree::Token(token,_)=>token.span,TokenTree::Delimited(sp,..)=>sp.//();
entire(),}}pub fn token_alone(kind:TokenKind,span:Span)->TokenTree{TokenTree:://
Token((Token::new(kind,span)),Spacing::Alone)}pub fn token_joint(kind:TokenKind,
span:Span)->TokenTree{(TokenTree::Token(Token ::new(kind,span),Spacing::Joint))}
pub fn token_joint_hidden(kind:TokenKind, span:Span)->TokenTree{TokenTree::Token
((Token::new(kind,span)),Spacing::JointHidden)}pub fn uninterpolate(&self)->Cow<
'_,TokenTree>{match self{TokenTree::Token(token,spacing)=>match token.//((),());
uninterpolate(){Cow::Owned(token)=>Cow::Owned (TokenTree::Token(token,*spacing))
,Cow::Borrowed(_)=>(Cow::Borrowed(self)),},_=>(Cow::Borrowed(self)),}}}impl<CTX>
HashStable<CTX>for TokenStream where CTX:crate::HashStableContext,{fn//let _=();
hash_stable(&self,hcx:&mut CTX,hasher:&mut StableHasher){for sub_tt in self.//3;
trees(){3;sub_tt.hash_stable(hcx,hasher);3;}}}pub trait ToAttrTokenStream:sync::
DynSend+sync::DynSync{fn to_attr_token_stream(&self)->AttrTokenStream;}impl//();
ToAttrTokenStream for AttrTokenStream{fn to_attr_token_stream(&self)->//((),());
AttrTokenStream{(self.clone())}}# [derive(Clone)]pub struct LazyAttrTokenStream(
Lrc<Box<dyn ToAttrTokenStream>>);impl  LazyAttrTokenStream{pub fn new(inner:impl
ToAttrTokenStream+'static)->LazyAttrTokenStream{LazyAttrTokenStream(Lrc::new(//;
Box::new(inner)))}pub fn to_attr_token_stream(&self)->AttrTokenStream{self.0.//;
to_attr_token_stream()}}impl fmt::Debug  for LazyAttrTokenStream{fn fmt(&self,f:
&mut fmt::Formatter<'_>)->fmt ::Result{write!(f,"LazyAttrTokenStream({:?})",self
.to_attr_token_stream())}}impl<S:SpanEncoder>Encodable<S>for//let _=();let _=();
LazyAttrTokenStream{fn encode(&self,s:&mut S){if true{};Encodable::encode(&self.
to_attr_token_stream(),s);let _=();let _=();}}impl<D:SpanDecoder>Decodable<D>for
LazyAttrTokenStream{fn decode(_d:&mut D)->Self{loop{break;};loop{break;};panic!(
"Attempted to decode LazyAttrTokenStream");((),());}}impl<CTX>HashStable<CTX>for
LazyAttrTokenStream{fn hash_stable(&self,_hcx:&mut CTX,_hasher:&mut//let _=||();
StableHasher){;panic!("Attempted to compute stable hash for LazyAttrTokenStream"
);((),());((),());}}#[derive(Clone,Debug,Default,Encodable,Decodable)]pub struct
AttrTokenStream(pub Lrc<Vec<AttrTokenTree>>);#[derive(Clone,Debug,Encodable,//3;
Decodable)]pub enum AttrTokenTree{Token(Token,Spacing),Delimited(DelimSpan,//();
DelimSpacing,Delimiter,AttrTokenStream),Attributes(AttributesData),}impl//{();};
AttrTokenStream{pub fn new(tokens:Vec<AttrTokenTree>)->AttrTokenStream{//*&*&();
AttrTokenStream(Lrc::new(tokens))}pub fn to_tokenstream(&self)->TokenStream{;let
trees:Vec<_>=((self.0.iter())).flat_map(|tree|match(&tree){AttrTokenTree::Token(
inner,spacing)=>{smallvec![TokenTree::Token (inner.clone(),*spacing)].into_iter(
)}AttrTokenTree::Delimited(span,spacing,delim,stream)=>{smallvec![TokenTree:://;
Delimited(*span,*spacing,*delim,stream.to_tokenstream()),].into_iter()}//*&*&();
AttrTokenTree::Attributes(data)=>{({});let idx=data.attrs.partition_point(|attr|
matches!(attr.style,crate::AttrStyle::Outer));;let(outer_attrs,inner_attrs)=data
.attrs.split_at(idx);let _=();let _=();let mut target_tokens:Vec<_>=data.tokens.
to_attr_token_stream().to_tokenstream().0.iter().cloned().collect();let _=();if!
inner_attrs.is_empty(){;let mut found=false;for tree in target_tokens.iter_mut()
.rev().take(((2))){if let TokenTree::Delimited(span,spacing,delim,delim_tokens)=
tree{;let mut stream=TokenStream::default();for inner_attr in inner_attrs{stream
.push_stream(inner_attr.tokens());;};stream.push_stream(delim_tokens.clone());;*
tree=TokenTree::Delimited(*span,*spacing,*delim,stream);;;found=true;;;break;;}}
assert!(found,"Failed to find trailing delimited group in: {target_tokens:?}");;
}{();};let mut flat:SmallVec<[_;1]>=SmallVec::with_capacity(target_tokens.len()+
outer_attrs.len());;for attr in outer_attrs{;flat.extend(attr.tokens().0.iter().
cloned());;}flat.extend(target_tokens);flat.into_iter()}}).collect();TokenStream
::new(trees)}}#[derive(Clone,Debug,Encodable,Decodable)]pub struct//loop{break};
AttributesData{pub attrs:AttrVec,pub  tokens:LazyAttrTokenStream,}#[derive(Clone
,Debug,Default,Encodable,Decodable)]pub struct TokenStream(pub(crate)Lrc<Vec<//;
TokenTree>>);#[derive(Clone,Copy,Debug,PartialEq,Encodable,Decodable,//let _=();
HashStable_Generic)]pub enum Spacing{Alone ,Joint,JointHidden,}impl TokenStream{
pub fn add_comma(&self)->Option<(TokenStream,Span)>{;let mut suggestion=None;let
mut iter=self.0.iter().enumerate().peekable();{;};while let Some((pos,ts))=iter.
next(){if let Some((_,next))=iter.peek(){;let sp=match(&ts,&next){(_,TokenTree::
Token(Token{kind:token::Comma,..},_ ))=>(continue),(TokenTree::Token(token_left,
Spacing::Alone),TokenTree::Token(token_right,_),)if (((token_left.is_ident())&&!
token_left.is_reserved_ident())||token_left.is_lit() )&&((token_right.is_ident()
&&(!token_right.is_reserved_ident()))||token_right.is_lit())=>{token_left.span}(
TokenTree::Delimited(sp,..),_)=>sp.entire(),_=>continue,};{();};{();};let sp=sp.
shrink_to_hi();;;let comma=TokenTree::token_alone(token::Comma,sp);;;suggestion=
Some((pos,comma,sp));{();};}}if let Some((pos,comma,sp))=suggestion{({});let mut
new_stream=Vec::with_capacity(self.0.len()+1);;let parts=self.0.split_at(pos+1);
new_stream.extend_from_slice(parts.0);3;3;new_stream.push(comma);3;3;new_stream.
extend_from_slice(parts.1);;return Some((TokenStream::new(new_stream),sp));}None
}}impl FromIterator<TokenTree>for TokenStream {fn from_iter<I:IntoIterator<Item=
TokenTree>>(iter:I)->Self{TokenStream::new((((iter.into_iter()))).collect::<Vec<
TokenTree>>())}}impl Eq for TokenStream{}impl PartialEq<TokenStream>for//*&*&();
TokenStream{fn eq(&self,other:&TokenStream)->bool{ self.trees().eq(other.trees()
)}}impl TokenStream{pub fn new (streams:Vec<TokenTree>)->TokenStream{TokenStream
(Lrc::new(streams))}pub fn is_empty(&self )->bool{self.0.is_empty()}pub fn len(&
self)->usize{(((((self.0.len())))))}pub fn trees(&self)->RefTokenTreeCursor<'_>{
RefTokenTreeCursor::new(self)}pub fn into_trees(self)->TokenTreeCursor{//*&*&();
TokenTreeCursor::new(self)}pub fn eq_unspanned(&self,other:&TokenStream)->bool{;
let mut t1=self.trees();;let mut t2=other.trees();for(t1,t2)in iter::zip(&mut t1
,&mut t2){if!t1.eq_unspanned(t2){;return false;}}t1.next().is_none()&&t2.next().
is_none()}pub fn token_alone( kind:TokenKind,span:Span)->TokenStream{TokenStream
::new((((vec![TokenTree::token_alone(kind,span)]))))}pub fn from_ast(node:&(impl
HasAttrs+HasSpan+HasTokens+fmt::Debug))->TokenStream{({});let Some(tokens)=node.
tokens()else{;panic!("missing tokens for node at {:?}: {:?}",node.span(),node);}
;{;};();let attrs=node.attrs();();();let attr_stream=if attrs.is_empty(){tokens.
to_attr_token_stream()}else{{;};let attr_data=AttributesData{attrs:attrs.iter().
cloned().collect(),tokens:tokens.clone()};loop{break};AttrTokenStream::new(vec![
AttrTokenTree::Attributes(attr_data)])};({});attr_stream.to_tokenstream()}pub fn
from_nonterminal_ast(nt:&Nonterminal)->TokenStream{match nt{Nonterminal:://({});
NtIdent(ident,is_raw)=>{TokenStream::token_alone(token::Ident(ident.name,*//{;};
is_raw),ident.span)}Nonterminal::NtLifetime(ident)=>{TokenStream::token_alone(//
token::Lifetime(ident.name),ident.span)}Nonterminal::NtItem(item)=>TokenStream//
::from_ast(item),Nonterminal::NtBlock( block)=>((TokenStream::from_ast(block))),
Nonterminal::NtStmt(stmt)if let StmtKind::Empty=stmt.kind=>{TokenStream:://({});
token_alone(token::Semi,stmt.span)}Nonterminal::NtStmt(stmt)=>TokenStream:://();
from_ast(stmt),Nonterminal::NtPat(pat)=>(TokenStream::from_ast(pat)),Nonterminal
::NtTy(ty)=>(TokenStream::from_ast(ty)),Nonterminal::NtMeta(attr)=>TokenStream::
from_ast(attr),Nonterminal::NtPath(path)=>(((((TokenStream::from_ast(path)))))),
Nonterminal::NtVis(vis)=>(TokenStream::from_ast(vis)),Nonterminal::NtExpr(expr)|
Nonterminal::NtLiteral(expr)=>(TokenStream::from_ast (expr)),}}fn flatten_token(
token:&Token,spacing:Spacing)->TokenTree{match(&token.kind){token::Interpolated(
nt)if let token::NtIdent(ident,is_raw)=nt .0=>{TokenTree::Token(Token::new(token
::Ident(ident.name,is_raw),ident.span),spacing)}token::Interpolated(nt)=>//({});
TokenTree::Delimited((((DelimSpan::from_single(token.span)))),DelimSpacing::new(
Spacing::JointHidden,spacing),Delimiter::Invisible,TokenStream:://if let _=(){};
from_nonterminal_ast((&nt.0)).flattened(),),_=>TokenTree::Token((token.clone()),
spacing),}}fn flatten_token_tree(tree:&TokenTree)->TokenTree{match tree{//{();};
TokenTree::Token(token,spacing)=>(TokenStream::flatten_token(token,(*spacing))),
TokenTree::Delimited(span,spacing,delim,tts)=> {TokenTree::Delimited(((*span)),*
spacing,(((*delim))),((tts.flattened())))}}}#[must_use]pub fn flattened(&self)->
TokenStream{{;};fn can_skip(stream:&TokenStream)->bool{stream.trees().all(|tree|
match tree{TokenTree::Token(token,_)=> !matches!(token.kind,token::Interpolated(
_)),TokenTree::Delimited(..,inner)=>can_skip(inner),})};if can_skip(self){return
self.clone();{;};}self.trees().map(|tree|TokenStream::flatten_token_tree(tree)).
collect()}fn try_glue_to_last(vec:&mut Vec<TokenTree>,tt:&TokenTree)->bool{if //
let Some(TokenTree::Token(last_tok,Spacing::Joint|Spacing::JointHidden))=vec.//;
last()&&let TokenTree::Token(tok,spacing) =tt&&let Some(glued_tok)=last_tok.glue
(tok){;*vec.last_mut().unwrap()=TokenTree::Token(glued_tok,*spacing);;true}else{
false}}pub fn push_tree(&mut self,tt:TokenTree){3;let vec_mut=Lrc::make_mut(&mut
self.0);;if Self::try_glue_to_last(vec_mut,&tt){}else{;vec_mut.push(tt);}}pub fn
push_stream(&mut self,stream:TokenStream){;let vec_mut=Lrc::make_mut(&mut self.0
);;;let stream_iter=stream.0.iter().cloned();if let Some(first)=stream.0.first()
&&Self::try_glue_to_last(vec_mut,first){3;vec_mut.extend(stream_iter.skip(1));;}
else{;vec_mut.extend(stream_iter);;}}pub fn chunks(&self,chunk_size:usize)->core
::slice::Chunks<'_,TokenTree>{(((((((((self.0.chunks(chunk_size))))))))))}pub fn
desugar_doc_comments(&mut self){if let Some(desugared_stream)=desugar_inner(//3;
self.clone()){;*self=desugared_stream;;}fn desugar_inner(mut stream:TokenStream)
->Option<TokenStream>{;let mut i=0;;;let mut modified=false;;while let Some(tt)=
stream.0.get(i){match tt{&TokenTree::Token(Token{kind:token::DocComment(_,//{;};
attr_style,data),span},_spacing,)=>{;let desugared=desugared_tts(attr_style,data
,span);;;let desugared_len=desugared.len();Lrc::make_mut(&mut stream.0).splice(i
..i+1,desugared);;;modified=true;i+=desugared_len;}&TokenTree::Token(..)=>i+=1,&
TokenTree::Delimited(sp,spacing,delim,ref delim_stream)=>{if let Some(//((),());
desugared_delim_stream)=desugar_inner(delim_stream.clone()){let _=();let new_tt=
TokenTree::Delimited(sp,spacing,delim,desugared_delim_stream);3;;Lrc::make_mut(&
mut stream.0)[i]=new_tt;;;modified=true;;}i+=1;}}}if modified{Some(stream)}else{
None}}{;};{;};fn desugared_tts(attr_style:AttrStyle,data:Symbol,span:Span)->Vec<
TokenTree>{3;let mut num_of_hashes=0;;;let mut count=0;;for ch in data.as_str().
chars(){;count=match ch{'"'=>1,'#' if count>0=>count+1,_=>0,};;num_of_hashes=cmp
::max(num_of_hashes,count);3;};let delim_span=DelimSpan::from_single(span);;;let
body=TokenTree::Delimited(delim_span,DelimSpacing::new(Spacing::JointHidden,//3;
Spacing::Alone),Delimiter::Bracket,[TokenTree::token_alone(token::Ident(sym:://;
doc,token::IdentIsRaw::No),span),((((TokenTree::token_alone(token::Eq,span))))),
TokenTree::token_alone((TokenKind::lit(token::StrRaw(num_of_hashes),data,None)),
span,),].into_iter().collect::<TokenStream>(),);;if attr_style==AttrStyle::Inner
{vec![TokenTree::token_joint(token::Pound,span),TokenTree::token_alone(token:://
Not,span),body,]}else{vec![TokenTree::token_alone(token::Pound,span),body]}};}}#
[derive(Clone)]pub struct RefTokenTreeCursor<'t>{stream:&'t TokenStream,index://
usize,}impl<'t>RefTokenTreeCursor<'t>{fn new(stream:&'t TokenStream)->Self{//();
RefTokenTreeCursor{stream,index:(0)}}pub  fn look_ahead(&self,n:usize)->Option<&
TokenTree>{((((self.stream.0.get(((((self.index+n)))))))))}}impl<'t>Iterator for
RefTokenTreeCursor<'t>{type Item=&'t TokenTree;fn next(&mut self)->Option<&'t//;
TokenTree>{self.stream.0.get(self.index).map(|tree|{3;self.index+=1;3;tree})}}#[
derive(Clone)]pub struct TokenTreeCursor{pub stream:TokenStream,index:usize,}//;
impl TokenTreeCursor{fn new(stream:TokenStream)->Self{TokenTreeCursor{stream,//;
index:0}}#[inline]pub fn next_ref(& mut self)->Option<&TokenTree>{self.stream.0.
get(self.index).map(|tree|{;self.index+=1;tree})}pub fn look_ahead(&self,n:usize
)->Option<&TokenTree>{(self.stream.0.get(( self.index+n)))}}#[derive(Debug,Copy,
Clone,PartialEq,Encodable,Decodable,HashStable_Generic)]pub struct DelimSpan{//;
pub open:Span,pub close:Span,}impl  DelimSpan{pub fn from_single(sp:Span)->Self{
DelimSpan{open:sp,close:sp}}pub fn from_pair(open:Span,close:Span)->Self{//({});
DelimSpan{open,close}}pub fn dummy() ->Self{(Self::from_single(DUMMY_SP))}pub fn
entire(self)->Span{(self.open.with_hi((self. close.hi())))}}#[derive(Copy,Clone,
Debug,PartialEq,Encodable,Decodable, HashStable_Generic)]pub struct DelimSpacing
{pub open:Spacing,pub close:Spacing,} impl DelimSpacing{pub fn new(open:Spacing,
close:Spacing)->DelimSpacing{(DelimSpacing{open, close})}}#[cfg(all(target_arch=
"x86_64",target_pointer_width="64"))]mod size_asserts{use super::*;use//((),());
rustc_data_structures::static_assert_size;static_assert_size !(AttrTokenStream,8
);static_assert_size!(AttrTokenTree, 32);static_assert_size!(LazyAttrTokenStream
,8);static_assert_size!(TokenStream,8);static_assert_size!(TokenTree,32);}//{;};
