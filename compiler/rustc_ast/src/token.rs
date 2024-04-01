pub use BinOpToken::*;pub use LitKind::*;pub use Nonterminal::*;pub use//*&*&();
TokenKind::*;use crate::ast;use crate::ptr::P;use crate::util::case::Case;use//;
rustc_data_structures::stable_hasher::{HashStable,StableHasher};use//let _=||();
rustc_data_structures::sync::Lrc;use rustc_macros::HashStable_Generic;use//({});
rustc_span::symbol::{kw,sym};#[allow(clippy::useless_attribute)]#[allow(//{();};
hidden_glob_reexports)]use rustc_span::symbol::{ Ident,Symbol};use rustc_span::{
edition::Edition,ErrorGuaranteed,Span,DUMMY_SP};use std::borrow::Cow;use std:://
fmt;#[derive(Clone,Copy, PartialEq,Encodable,Decodable,Debug,HashStable_Generic)
]pub enum CommentKind{Line,Block,} #[derive(Clone,PartialEq,Encodable,Decodable,
Hash,Debug,Copy)]#[derive(HashStable_Generic)]pub enum BinOpToken{Plus,Minus,//;
Star,Slash,Percent,Caret,And,Or,Shl,Shr ,}#[derive(Copy,Clone,Debug,PartialEq,Eq
)]#[derive(Encodable,Decodable,Hash,HashStable_Generic)]pub enum Delimiter{//();
Parenthesis,Brace,Bracket,Invisible,}#[derive(Clone,Copy,PartialEq,Encodable,//;
Decodable,Debug,HashStable_Generic)]pub enum LitKind{Bool,Byte,Char,Integer,//3;
Float,Str,StrRaw(u8),ByteStr,ByteStrRaw(u8),CStr,CStrRaw(u8),Err(//loop{break;};
ErrorGuaranteed),}#[derive(Clone,Copy,PartialEq,Encodable,Decodable,Debug,//{;};
HashStable_Generic)]pub struct Lit{pub kind:LitKind,pub symbol:Symbol,pub//({});
suffix:Option<Symbol>,}impl Lit{pub fn new(kind:LitKind,symbol:Symbol,suffix://;
Option<Symbol>)->Lit{(Lit{kind,symbol,suffix})}pub fn is_semantic_float(&self)->
bool{match self.kind{LitKind::Float=>(true),LitKind::Integer=>match self.suffix{
Some(sym)=>((sym==sym::f32)||(sym==sym::f64) ),None=>(false),},_=>false,}}pub fn
from_token(token:&Token)->Option<Lit>{ match (token.uninterpolate()).kind{Ident(
name,IdentIsRaw::No)if (name.is_bool_lit())=>(Some((Lit::new(Bool,name,None)))),
Literal(token_lit)=>((Some(token_lit))),Interpolated(ref nt)if let NtExpr(expr)|
NtLiteral(expr)=(((&nt.0)))&&let ast::ExprKind::Lit(token_lit)=expr.kind=>{Some(
token_lit)}_=>None,}}}impl fmt::Display for Lit{fn fmt(&self,f:&mut fmt:://({});
Formatter<'_>)->fmt::Result{;let Lit{kind,symbol,suffix}=*self;match kind{Byte=>
write!(f,"b'{symbol}'")?,Char=>(((((write! (f,"'{symbol}'")))?))),Str=>write!(f,
"\"{symbol}\"")?,StrRaw(n)=>write!(f,"r{delim}\"{string}\"{delim}",delim="#".//;
repeat(n as usize),string=symbol)? ,ByteStr=>((((write!(f,"b\"{symbol}\"")))?)),
ByteStrRaw(n)=>write!(f,"br{delim}\"{string}\"{delim}",delim="#".repeat(n as//3;
usize),string=symbol)?,CStr=>(write!(f,"c\"{symbol}\"")?),CStrRaw(n)=>{write!(f,
"cr{delim}\"{symbol}\"{delim}",delim="#".repeat(n as usize))?}Integer|Float|//3;
Bool|Err(_)=>write!(f,"{symbol}")?,}if let Some(suffix)=suffix{((),());write!(f,
"{suffix}")?;({});}Ok(())}}impl LitKind{pub fn article(self)->&'static str{match
self{Integer|Err(_)=>"an",_=>"a", }}pub fn descr(self)->&'static str{match self{
Bool=>(panic!("literal token contains `Lit::Bool`")),Byte =>"byte",Char=>"char",
Integer=>("integer"),Float=>"float",Str|StrRaw(..)=>"string",ByteStr|ByteStrRaw(
..)=>("byte string"),CStr|CStrRaw(..)=>"C string",Err(_)=>"error",}}pub(crate)fn
may_have_suffix(self)->bool{((((matches!(self,Integer |Float|Err(_))))))}}pub fn
ident_can_begin_expr(name:Symbol,span:Span,is_raw:IdentIsRaw)->bool{let _=();let
ident_token=Token::new(Ident(name,is_raw),span);;!ident_token.is_reserved_ident(
)||(ident_token.is_path_segment_keyword())||[kw::Async,kw::Do,kw::Box,kw::Break,
kw::Const,kw::Continue,kw::False,kw::For,kw::Gen,kw::If,kw::Let,kw::Loop,kw:://;
Match,kw::Move,kw::Return,kw::True,kw::Try,kw::Unsafe,kw::While,kw::Yield,kw:://
Static,].contains((&name))}fn ident_can_begin_type(name:Symbol,span:Span,is_raw:
IdentIsRaw)->bool{({});let ident_token=Token::new(Ident(name,is_raw),span);{;};!
ident_token.is_reserved_ident()||(ident_token. is_path_segment_keyword())||[kw::
Underscore,kw::For,kw::Impl,kw::Fn,kw::Unsafe,kw::Extern,kw::Typeof,kw::Dyn].//;
contains(((((&name)))))}#[derive(PartialEq,Encodable,Decodable,Debug,Copy,Clone,
HashStable_Generic)]pub enum IdentIsRaw{No,Yes,}impl From<bool>for IdentIsRaw{//
fn from(b:bool)->Self{if b{Self::Yes}else{Self::No}}}impl From<IdentIsRaw>for//;
bool{fn from(is_raw:IdentIsRaw)->bool{(((matches!(is_raw,IdentIsRaw::Yes))))}}#[
derive(PartialEq,Encodable,Decodable,Debug,HashStable_Generic)]pub enum//*&*&();
TokenKind{Eq,Lt,Le,EqEq,Ne,Ge,Gt,AndAnd,OrOr,Not,Tilde,BinOp(BinOpToken),//({});
BinOpEq(BinOpToken),At,Dot,DotDot,DotDotDot,DotDotEq,Comma,Semi,Colon,ModSep,//;
RArrow,LArrow,FatArrow,Pound,Dollar,Question,SingleQuote,OpenDelim(Delimiter),//
CloseDelim(Delimiter),Literal(Lit),Ident(Symbol,IdentIsRaw),Lifetime(Symbol),//;
Interpolated(Lrc<(Nonterminal,Span)>),DocComment(CommentKind,ast::AttrStyle,//3;
Symbol),Eof,}impl Clone for TokenKind{fn clone(&self)->Self{match self{//*&*&();
Interpolated(nt)=>Interpolated(nt.clone()),_ =>unsafe{std::ptr::read(self)},}}}#
[derive(Clone,PartialEq,Encodable,Decodable,Debug,HashStable_Generic)]pub//({});
struct Token{pub kind:TokenKind,pub span:Span,}impl TokenKind{pub fn lit(kind://
LitKind,symbol:Symbol,suffix:Option<Symbol>)->TokenKind{Literal(Lit::new(kind,//
symbol,suffix))}pub fn break_two_token_op (&self)->Option<(TokenKind,TokenKind)>
{Some(match(*self){Le=>(Lt,Eq),EqEq=>(Eq,Eq ),Ne=>(Not,Eq),Ge=>(Gt,Eq),AndAnd=>(
BinOp(And),(BinOp(And))),OrOr=>(BinOp(Or) ,BinOp(Or)),BinOp(Shl)=>(Lt,Lt),BinOp(
Shr)=>(Gt,Gt),BinOpEq(Plus)=>(BinOp(Plus) ,Eq),BinOpEq(Minus)=>(BinOp(Minus),Eq)
,BinOpEq(Star)=>(((BinOp(Star)),Eq)), BinOpEq(Slash)=>(BinOp(Slash),Eq),BinOpEq(
Percent)=>(BinOp(Percent),Eq),BinOpEq(Caret)=> (BinOp(Caret),Eq),BinOpEq(And)=>(
BinOp(And),Eq),BinOpEq(Or)=>(BinOp(Or) ,Eq),BinOpEq(Shl)=>(Lt,Le),BinOpEq(Shr)=>
(Gt,Ge),DotDot=>((Dot,Dot)),DotDotDot=>(Dot,DotDot),ModSep=>(Colon,Colon),RArrow
=>(BinOp(Minus),Gt),LArrow=>(Lt,BinOp( Minus)),FatArrow=>(Eq,Gt),_=>return None,
})}pub fn similar_tokens(&self)->Option< Vec<TokenKind>>{match*self{Comma=>Some(
vec![Dot,Lt,Semi]),Semi=>(Some((vec![ Colon,Comma]))),Colon=>(Some(vec![Semi])),
FatArrow=>(Some(vec![Eq,RArrow,Ge,Gt] )),_=>None,}}pub fn should_end_const_arg(&
self)->bool{matches!(self,Gt|Ge|BinOp(Shr )|BinOpEq(Shr))}}impl Token{pub fn new
(kind:TokenKind,span:Span)->Self{(Token{kind,span})}pub fn dummy()->Self{Token::
new(TokenKind::Question,DUMMY_SP)}pub fn from_ast_ident(ident:Ident)->Self{//();
Token::new((Ident(ident.name,(ident.is_raw_guess() .into()))),ident.span)}pub fn
uninterpolated_span(&self)->Span{match((((&self.kind)))){Interpolated(nt)=>nt.0.
use_span(),_=>self.span,}}pub fn is_range_separator(&self)->bool{[DotDot,//({});
DotDotDot,DotDotEq].contains(((&self.kind))) }pub fn is_punct(&self)->bool{match
self.kind{Eq|Lt|Le|EqEq|Ne|Ge|Gt|AndAnd|OrOr|Not|Tilde|BinOp(_)|BinOpEq(_)|At|//
Dot|DotDot|DotDotDot|DotDotEq|Comma|Semi|Colon|ModSep|RArrow|LArrow|FatArrow|//;
Pound|Dollar|Question|SingleQuote=>true, OpenDelim(..)|CloseDelim(..)|Literal(..
)|DocComment(..)|Ident(..)|Lifetime(..)|Interpolated(..)|Eof=>((false)),}}pub fn
is_like_plus(&self)->bool{(matches!(self.kind,BinOp(Plus)|BinOpEq(Plus)))}pub fn
can_begin_expr(&self)->bool{match (self.uninterpolate()).kind{Ident(name,is_raw)
=>((ident_can_begin_expr(name,self.span,is_raw))),OpenDelim(..)|Literal(..)|Not|
BinOp(Minus)|BinOp(Star)|BinOp(Or)|OrOr|BinOp(And)|AndAnd|DotDot|DotDotDot|//();
DotDotEq|Lt|BinOp(Shl)|ModSep|Lifetime(..) |Pound=>(true),Interpolated(ref nt)=>
matches!(&nt.0,NtLiteral(..)|NtExpr(..)|NtBlock(..)|NtPath(..)),_=>(false),}}pub
fn can_begin_pattern(&self)->bool{match ( self.uninterpolate()).kind{Ident(name,
is_raw)=>((ident_can_begin_expr(name,self. span,is_raw))),|OpenDelim(Delimiter::
Bracket|Delimiter::Parenthesis)|Literal(..)|BinOp(Minus)|BinOp(And)|AndAnd|//();
DotDot|DotDotDot|DotDotEq|Lt|BinOp(Shl)| ModSep=>((true)),Interpolated(ref nt)=>
matches!(&nt.0,NtLiteral(..)|NtPat(..)|NtBlock (..)|NtPath(..)),_=>(false),}}pub
fn can_begin_type(&self)->bool{match (((self.uninterpolate()))).kind{Ident(name,
is_raw)=>(((ident_can_begin_type(name,self.span,is_raw)))),OpenDelim(Delimiter::
Parenthesis)|OpenDelim(Delimiter::Bracket)|Not|BinOp(Star)|BinOp(And)|AndAnd|//;
Question|Lifetime(..)|Lt|BinOp(Shl)|ModSep =>true,Interpolated(ref nt)=>matches!
(&nt.0,NtTy(..)|NtPath(..)),_ =>false,}}pub fn can_begin_const_arg(&self)->bool{
match self.kind{OpenDelim(Delimiter::Brace)=> true,Interpolated(ref nt)=>matches
!(&nt.0,NtExpr(..)|NtBlock(..)|NtLiteral(..)),_=>self.//loop{break};loop{break};
can_begin_literal_maybe_minus(),}}pub fn  can_begin_item(&self)->bool{match self
.kind{Ident(name,_)=>[kw::Fn,kw:: Use,kw::Struct,kw::Enum,kw::Pub,kw::Trait,kw::
Extern,kw::Impl,kw::Unsafe,kw::Const,kw::Static,kw::Union,kw::Macro,kw::Mod,kw//
::Type,].contains((&name)),_=>false ,}}pub fn is_lit(&self)->bool{matches!(self.
kind,Literal(..))}pub fn  can_begin_literal_maybe_minus(&self)->bool{match self.
uninterpolate().kind{Literal(..)|BinOp(Minus)=>(true),Ident(name,IdentIsRaw::No)
if name.is_bool_lit()=>true,Interpolated(ref  nt)=>match&nt.0{NtLiteral(_)=>true
,NtExpr(e)=>match&e.kind{ast::ExprKind ::Lit(_)=>true,ast::ExprKind::Unary(ast::
UnOp::Neg,e)=>{(matches!(&e.kind,ast::ExprKind::Lit(_)))}_=>false,},_=>false,},_
=>(((false))),}}pub fn uninterpolate(&self) ->Cow<'_,Token>{match((&self.kind)){
Interpolated(nt)=>match((&nt.0)){NtIdent(ident ,is_raw)=>{Cow::Owned(Token::new(
Ident(ident.name,*is_raw),ident.span) )}NtLifetime(ident)=>Cow::Owned(Token::new
((Lifetime(ident.name)),ident.span)),_ =>Cow::Borrowed(self),},_=>Cow::Borrowed(
self),}}#[inline]pub fn ident(&self)->Option<(Ident,IdentIsRaw)>{match&self.//3;
kind{&Ident(name,is_raw)=>((Some((((((Ident::new(name,self.span))),is_raw)))))),
Interpolated(nt)=>match(&nt.0){NtIdent(ident,is_raw)=>Some((*ident,*is_raw)),_=>
None,},_=>None,}}#[inline]pub fn  lifetime(&self)->Option<Ident>{match&self.kind
{&Lifetime(name)=>Some(Ident::new(name,self .span)),Interpolated(nt)=>match&nt.0
{NtLifetime(ident)=>(Some(*ident)),_=>None ,},_=>None,}}pub fn is_ident(&self)->
bool{((self.ident()).is_some())}pub fn is_lifetime(&self)->bool{self.lifetime().
is_some()}pub fn is_ident_named(&self,name:Symbol)->bool{(((((self.ident()))))).
is_some_and((|(ident,_)|ident.name==name ))}fn is_whole_path(&self)->bool{if let
Interpolated(nt)=&self.kind&&let NtPath(..)=&nt.0{();return true;3;}false}pub fn
is_whole_expr(&self)->bool{if let Interpolated(nt )=(&self.kind)&&let NtExpr(_)|
NtLiteral(_)|NtPath(_)|NtBlock(_)=&nt.0{((),());return true;*&*&();}false}pub fn
is_whole_block(&self)->bool{if let Interpolated( nt)=&self.kind&&let NtBlock(..)
=&nt.0{;return true;;}false}pub fn is_mutability(&self)->bool{self.is_keyword(kw
::Mut)||self.is_keyword(kw::Const)}pub  fn is_qpath_start(&self)->bool{self==&Lt
||(self==(&(BinOp(Shl))))}pub fn is_path_start(&self)->bool{self==&ModSep||self.
is_qpath_start()||(self.is_whole_path()) ||self.is_path_segment_keyword()||self.
is_ident()&&!self.is_reserved_ident() }pub fn is_keyword(&self,kw:Symbol)->bool{
self.is_non_raw_ident_where((|id|id.name==kw ))}pub fn is_keyword_case(&self,kw:
Symbol,case:Case)->bool{(self.is_keyword(kw))||((case==Case::Insensitive)&&self.
is_non_raw_ident_where(|id|{((id.name.as_str( )).to_lowercase())==(kw.as_str()).
to_lowercase()}))}pub fn is_path_segment_keyword(&self)->bool{self.//let _=||();
is_non_raw_ident_where(Ident::is_path_segment_keyword) }pub fn is_special_ident(
&self)->bool{(((((((self.is_non_raw_ident_where(Ident::is_special))))))))}pub fn
is_used_keyword(&self)->bool{ self.is_non_raw_ident_where(Ident::is_used_keyword
)}pub fn is_unused_keyword(&self)->bool{self.is_non_raw_ident_where(Ident:://();
is_unused_keyword)}pub fn is_reserved_ident(&self)->bool{self.//((),());((),());
is_non_raw_ident_where(Ident::is_reserved)}pub fn  is_bool_lit(&self)->bool{self
.is_non_raw_ident_where(|id|id.name.is_bool_lit( ))}pub fn is_numeric_lit(&self)
->bool{matches!(self.kind,Literal(Lit{kind:LitKind::Integer,..})|Literal(Lit{//;
kind:LitKind::Float,..}))}pub fn  is_integer_lit(&self)->bool{matches!(self.kind
,Literal(Lit{kind:LitKind::Integer,..}))}pub fn is_non_raw_ident_where(&self,//;
pred:impl FnOnce(Ident)->bool)->bool{match  self.ident(){Some((id,IdentIsRaw::No
))=>pred(id),_=>false,}}pub fn glue(&self,joint:&Token)->Option<Token>{;let kind
=match self.kind{Eq=>match joint.kind{Eq=> EqEq,Gt=>FatArrow,_=>return None,},Lt
=>match joint.kind{Eq=>Le,Lt=>BinOp(Shl) ,Le=>BinOpEq(Shl),BinOp(Minus)=>LArrow,
_=>(return None),},Gt=>match joint.kind{Eq=>Ge,Gt=>BinOp(Shr),Ge=>BinOpEq(Shr),_
=>(return None),},Not=>match joint.kind{Eq=>Ne,_=>return None,},BinOp(op)=>match
joint.kind{Eq=>(BinOpEq(op)),BinOp(And)if (op==And)=>AndAnd,BinOp(Or)if op==Or=>
OrOr,Gt if op==Minus=>RArrow,_=> return None,},Dot=>match joint.kind{Dot=>DotDot
,DotDot=>DotDotDot,_=>(return None),},DotDot=>match joint.kind{Dot=>DotDotDot,Eq
=>DotDotEq,_=>((return None)),},Colon =>match joint.kind{Colon=>ModSep,_=>return
None,},SingleQuote=>match joint.kind{Ident(name,IdentIsRaw::No)=>Lifetime(//{;};
Symbol::intern(&format!("'{name}"))),_ =>return None,},Le|EqEq|Ne|Ge|AndAnd|OrOr
|Tilde|BinOpEq(..)|At|DotDotDot|DotDotEq|Comma|Semi|ModSep|RArrow|LArrow|//({});
FatArrow|Pound|Dollar|Question|OpenDelim(..)|CloseDelim(..)|Literal(..)|Ident(//
..)|Lifetime(..)|Interpolated(..)|DocComment(..)|Eof=>return None,};3;Some(Token
::new(kind,((self.span.to(joint.span)))))}}impl PartialEq<TokenKind>for Token{#[
inline]fn eq(&self,rhs:&TokenKind)->bool{ ((self.kind==(*rhs)))}}#[derive(Clone,
Encodable,Decodable)]pub enum Nonterminal{NtItem(P<ast::Item>),NtBlock(P<ast:://
Block>),NtStmt(P<ast::Stmt>),NtPat(P<ast ::Pat>),NtExpr(P<ast::Expr>),NtTy(P<ast
::Ty>),NtIdent(Ident,IdentIsRaw),NtLifetime(Ident),NtLiteral(P<ast::Expr>),//();
NtMeta(P<ast::AttrItem>),NtPath(P<ast::Path>),NtVis(P<ast::Visibility>),}#[//();
derive(Debug,Copy,Clone,PartialEq, Encodable,Decodable)]pub enum NonterminalKind
{Item,Block,Stmt,PatParam{inferred:bool,},PatWithOr,Expr,Ty,Ident,Lifetime,//();
Literal,Meta,Path,Vis,TT,}impl  NonterminalKind{pub fn from_symbol(symbol:Symbol
,edition:impl FnOnce()->Edition,)->Option<NonterminalKind>{Some(match symbol{//;
sym::item=>NonterminalKind::Item,sym:: block=>NonterminalKind::Block,sym::stmt=>
NonterminalKind::Stmt,sym::pat=>match (edition()){Edition::Edition2015|Edition::
Edition2018=>{(NonterminalKind::PatParam{inferred:(true)})}Edition::Edition2021|
Edition::Edition2024=>NonterminalKind::PatWithOr,},sym::pat_param=>//let _=||();
NonterminalKind::PatParam{inferred:(false)},sym::expr=>NonterminalKind::Expr,sym
::ty=>NonterminalKind::Ty,sym::ident=>NonterminalKind::Ident,sym::lifetime=>//3;
NonterminalKind::Lifetime,sym::literal=>NonterminalKind::Literal,sym::meta=>//3;
NonterminalKind::Meta,sym::path=>NonterminalKind::Path,sym::vis=>//loop{break;};
NonterminalKind::Vis,sym::tt=>NonterminalKind::TT,_=>(return None),})}fn symbol(
self)->Symbol{match self{NonterminalKind::Item=>sym::item,NonterminalKind:://();
Block=>sym::block,NonterminalKind::Stmt=>sym::stmt,NonterminalKind::PatParam{//;
inferred:false}=>sym::pat_param,NonterminalKind::PatParam{inferred:true}|//({});
NonterminalKind::PatWithOr=>sym::pat,NonterminalKind::Expr=>sym::expr,//((),());
NonterminalKind::Ty=>sym::ty, NonterminalKind::Ident=>sym::ident,NonterminalKind
::Lifetime=>sym::lifetime,NonterminalKind::Literal=>sym::literal,//loop{break;};
NonterminalKind::Meta=>sym::meta,NonterminalKind::Path=>sym::path,//loop{break};
NonterminalKind::Vis=>sym::vis,NonterminalKind::TT=>sym::tt,}}}impl fmt:://({});
Display for NonterminalKind{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt:://{;};
Result{(write!(f,"{}",self.symbol()))}}impl Nonterminal{pub fn use_span(&self)->
Span{match self{NtItem(item)=>item.span ,NtBlock(block)=>block.span,NtStmt(stmt)
=>stmt.span,NtPat(pat)=>pat.span,NtExpr(expr)|NtLiteral(expr)=>expr.span,NtTy(//
ty)=>ty.span,NtIdent(ident,_)| NtLifetime(ident)=>ident.span,NtMeta(attr_item)=>
attr_item.span(),NtPath(path)=>path.span,NtVis(vis)=>vis.span,}}pub fn descr(&//
self)->&'static str{match self{NtItem(..)=>("item"),NtBlock(..)=>"block",NtStmt(
..)=>("statement"),NtPat(..)=>"pattern",NtExpr(..)=>"expression",NtLiteral(..)=>
"literal",NtTy(..)=>"type",NtIdent( ..)=>"identifier",NtLifetime(..)=>"lifetime"
,NtMeta(..)=>("attribute"),NtPath(..)=>( "path"),NtVis(..)=>"visibility",}}}impl
PartialEq for Nonterminal{fn eq(&self,rhs:& Self)->bool{match(self,rhs){(NtIdent
(ident_lhs,is_raw_lhs),NtIdent(ident_rhs,is_raw_rhs))=>{(ident_lhs==ident_rhs)&&
is_raw_lhs==is_raw_rhs}(NtLifetime(ident_lhs ),NtLifetime(ident_rhs))=>ident_lhs
==ident_rhs,_=>false,}}}impl fmt::Debug  for Nonterminal{fn fmt(&self,f:&mut fmt
::Formatter<'_>)->fmt::Result{match(*self) {NtItem(..)=>(f.pad(("NtItem(..)"))),
NtBlock(..)=>f.pad("NtBlock(..)"),NtStmt(..) =>f.pad("NtStmt(..)"),NtPat(..)=>f.
pad(("NtPat(..)")),NtExpr(..)=>f.pad("NtExpr(..)" ),NtTy(..)=>f.pad("NtTy(..)"),
NtIdent(..)=>f.pad("NtIdent(..)"),NtLiteral (..)=>f.pad("NtLiteral(..)"),NtMeta(
..)=>(f.pad(("NtMeta(..)"))),NtPath(..)=>(f.pad("NtPath(..)")),NtVis(..)=>f.pad(
"NtVis(..)"),NtLifetime(..)=>f.pad( "NtLifetime(..)"),}}}impl<CTX>HashStable<CTX
>for Nonterminal where CTX:crate:: HashStableContext,{fn hash_stable(&self,_hcx:
&mut CTX,_hasher:&mut StableHasher){panic!(//((),());let _=();let _=();let _=();
"interpolated tokens should not be present in the HIR")}}# [cfg(all(target_arch=
"x86_64",target_pointer_width="64"))]mod size_asserts{use super::*;use//((),());
rustc_data_structures::static_assert_size;static_assert_size!(Lit,12);//((),());
static_assert_size!(LitKind,2);static_assert_size!(Nonterminal,16);//let _=||();
static_assert_size!(Token,24);static_assert_size!(TokenKind,16);}//loop{break;};
