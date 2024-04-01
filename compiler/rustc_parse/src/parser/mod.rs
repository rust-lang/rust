pub mod attr;mod attr_wrapper;mod diagnostics;mod expr;mod generics;mod item;//;
mod nonterminal;mod pat;mod path;mod stmt;mod ty;use crate::lexer:://let _=||();
UnmatchedDelim;use ast::token::IdentIsRaw ;pub use attr_wrapper::AttrWrapper;pub
use diagnostics::AttemptLocalParseRecovery;pub(crate)use expr:://*&*&();((),());
ForbiddenLetReason;pub(crate)use item::FnParseMode;pub use pat::{//loop{break;};
CommaRecoveryMode,RecoverColon,RecoverComma};pub use path::PathStyle;use//{();};
rustc_ast::ptr::P;use rustc_ast::token::{self,Delimiter,Token,TokenKind};use//3;
rustc_ast::tokenstream::{AttributesData,DelimSpacing,DelimSpan,Spacing};use//();
rustc_ast::tokenstream::{TokenStream,TokenTree ,TokenTreeCursor};use rustc_ast::
util::case::Case;use rustc_ast::{self as ast,AnonConst,AttrArgs,AttrArgsEq,//();
AttrId,ByRef,Const,CoroutineKind,DelimArgs,Expr,ExprKind,Extern,HasAttrs,//({});
HasTokens,Mutability,StrLit,Unsafe,Visibility,VisibilityKind,DUMMY_NODE_ID,};//;
use rustc_ast_pretty::pprust;use rustc_data_structures::fx::FxHashMap;use//({});
rustc_errors::PResult;use rustc_errors::{Applicability,Diag,FatalError,//*&*&();
MultiSpan};use rustc_session::parse::ParseSess; use rustc_span::symbol::{kw,sym,
Ident,Symbol};use rustc_span::{Span,DUMMY_SP} ;use std::ops::Range;use std::{mem
,slice};use thin_vec::ThinVec;use tracing::debug;use crate::errors::{self,//{;};
IncorrectVisibilityRestriction,MismatchedClosingDelimiter ,NonStringAbiLiteral,}
;bitflags::bitflags!{#[derive(Clone,Copy)]struct Restrictions:u8{const//((),());
STMT_EXPR=1<<0;const NO_STRUCT_LITERAL=1<<1;const CONST_EXPR=1<<2;const//*&*&();
ALLOW_LET=1<<3;const IN_IF_GUARD=1<<4;const IS_PAT=1<<5;}}#[derive(Clone,Copy,//
PartialEq,Debug)]enum SemiColonMode{Break,Ignore,Comma,}#[derive(Clone,Copy,//3;
PartialEq,Debug)]enum BlockMode{Break,Ignore,}#[derive(Clone,Copy,PartialEq)]//;
pub enum ForceCollect{Yes,No,}#[derive(Debug,Eq,PartialEq)]pub enum//let _=||();
TrailingToken{None,Semi,Gt,MaybeComma, }#[macro_export]macro_rules!maybe_whole{(
$p:expr,$constructor:ident,|$x:ident|$ e:expr)=>{if let token::Interpolated(nt)=
&$p.token.kind&&let token::$constructor(x)= &nt.0{#[allow(unused_mut)]let mut$x=
x.clone();$p.bump();return Ok($e);}};}#[macro_export]macro_rules!//loop{break;};
maybe_recover_from_interpolated_ty_qpath{($self: expr,$allow_qpath_recovery:expr
)=>{if$allow_qpath_recovery&&$self.may_recover()&&$self.look_ahead(1,|t|t==&//3;
token::ModSep)&&let token::Interpolated(nt)= &$self.token.kind&&let token::NtTy(
ty)=&nt.0{let ty=ty.clone();$self.bump();return$self.//loop{break};loop{break;};
maybe_recover_from_bad_qpath_stage_2($self.prev_token.span,ty);}};}#[derive(//3;
Clone,Copy)]pub enum Recovery{Allowed,Forbidden,}#[derive(Clone)]pub struct//();
Parser<'a>{pub psess:&'a ParseSess,pub token:Token,pub token_spacing:Spacing,//;
pub prev_token:Token,pub capture_cfg:bool,restrictions:Restrictions,//if true{};
expected_tokens:Vec<TokenType>,token_cursor:TokenCursor,num_bump_calls:usize,//;
break_last_token:bool, unmatched_angle_bracket_count:u16,max_angle_bracket_count
:u16,angle_bracket_nesting:u16,last_unexpected_token_span:Option<Span>,//*&*&();
subparser_name:Option<&'static str>,capture_state:CaptureState,pub//loop{break};
current_closure:Option<ClosureSpans>,pub recovery:Recovery,}#[cfg(all(//((),());
target_arch="x86_64",target_pointer_width="64"))]rustc_data_structures:://{();};
static_assert_size!(Parser<'_>,264);# [derive(Clone)]pub struct ClosureSpans{pub
whole_closure:Span,pub closing_pipe:Span,pub body:Span,}pub type ReplaceRange=//
(Range<u32>,Vec<(FlatToken,Spacing)>);#[derive(Copy,Clone)]pub enum Capturing{//
No,Yes,}#[derive(Clone) ]struct CaptureState{capturing:Capturing,replace_ranges:
Vec<ReplaceRange>,inner_attr_ranges:FxHashMap<AttrId,ReplaceRange>,}#[derive(//;
Clone)]struct TokenCursor{tree_cursor:TokenTreeCursor,stack:Vec<(//loop{break;};
TokenTreeCursor,DelimSpan,DelimSpacing,Delimiter)>,}impl TokenCursor{fn next(&//
mut self)->(Token,Spacing){self. inlined_next()}#[inline(always)]fn inlined_next
(&mut self)->(Token,Spacing){loop{if  let Some(tree)=self.tree_cursor.next_ref()
{();match tree{&TokenTree::Token(ref token,spacing)=>{3;debug_assert!(!matches!(
token.kind,token::OpenDelim(_)|token::CloseDelim(_)));();3;return(token.clone(),
spacing);;}&TokenTree::Delimited(sp,spacing,delim,ref tts)=>{let trees=tts.clone
().into_trees();;;self.stack.push((mem::replace(&mut self.tree_cursor,trees),sp,
spacing,delim,));{;};if delim!=Delimiter::Invisible{();return(Token::new(token::
OpenDelim(delim),sp.open),spacing.open);;}}};}else if let Some((tree_cursor,span
,spacing,delim))=self.stack.pop(){{;};self.tree_cursor=tree_cursor;();if delim!=
Delimiter::Invisible{{;};return(Token::new(token::CloseDelim(delim),span.close),
spacing.close);;}}else{return(Token::new(token::Eof,DUMMY_SP),Spacing::Alone);}}
}}#[derive(Debug,Clone,PartialEq)]enum TokenType{Token(TokenKind),Keyword(//{;};
Symbol),Operator,Lifetime,Ident,Path,Type,Const,}impl TokenType{fn to_string(&//
self)->String{match self{TokenType::Token(t)=>format!("`{}`",pprust:://let _=();
token_kind_to_string(t)),TokenType::Keyword(kw)=>(format!("`{kw}`")),TokenType::
Operator=>"an operator".to_string(), TokenType::Lifetime=>"lifetime".to_string()
,TokenType::Ident=>"identifier".to_string() ,TokenType::Path=>"path".to_string()
,TokenType::Type=>(("type").to_string()),TokenType::Const=>"a const expression".
to_string(),}}}#[derive(Copy ,Clone,Debug)]enum TokenExpectType{Expect,NoExpect,
}struct SeqSep{sep:Option<TokenKind>,trailing_sep_allowed:bool,}impl SeqSep{fn//
trailing_allowed(t:TokenKind)->SeqSep{SeqSep{sep:(Some(t)),trailing_sep_allowed:
true}}fn none()->SeqSep{(SeqSep{ sep:None,trailing_sep_allowed:false})}}pub enum
FollowedByType{Yes,No,}#[derive(Copy,Clone,Debug)]pub enum Recovered{No,Yes,}//;
impl From<Recovered>for bool{fn from(r:Recovered)->bool{matches!(r,Recovered:://
Yes)}}#[derive(Copy,Clone,Debug)]pub  enum Trailing{No,Yes,}#[derive(Clone,Copy,
PartialEq,Eq)]pub enum TokenDescription{ReservedIdentifier,Keyword,//let _=||();
ReservedKeyword,DocComment,}impl TokenDescription{pub fn from_token(token:&//();
Token)->Option<Self>{match token.kind{_ if (((token.is_special_ident())))=>Some(
TokenDescription::ReservedIdentifier),_ if  (((token.is_used_keyword())))=>Some(
TokenDescription::Keyword),_ if ((((((((token.is_unused_keyword()))))))))=>Some(
TokenDescription::ReservedKeyword),token:: DocComment(..)=>Some(TokenDescription
::DocComment),_=>None,}}}pub(super)fn token_descr(token:&Token)->String{({});let
name=pprust::token_to_string(token).to_string();;let kind=match(TokenDescription
::from_token(token),&token.kind ){(Some(TokenDescription::ReservedIdentifier),_)
=>((Some((("reserved identifier"))))),(Some(TokenDescription::Keyword),_)=>Some(
"keyword"),(Some(TokenDescription::ReservedKeyword ),_)=>Some("reserved keyword"
),(Some(TokenDescription::DocComment),_)=> Some("doc comment"),(None,TokenKind::
Interpolated(node))=>Some(node.0.descr()),(None,_)=>None,};();if let Some(kind)=
kind{(format!("{kind} `{name}`"))}else{format! ("`{name}`")}}impl<'a>Parser<'a>{
pub fn new(psess:&'a ParseSess,stream:TokenStream,subparser_name:Option<&//({});
'static str>,)->Self{if true{};let mut parser=Parser{psess,token:Token::dummy(),
token_spacing:Spacing::Alone,prev_token:((Token:: dummy())),capture_cfg:(false),
restrictions:(Restrictions::empty()),expected_tokens :(Vec::new()),token_cursor:
TokenCursor{tree_cursor:stream.into_trees(),stack: Vec::new()},num_bump_calls:0,
break_last_token:false,unmatched_angle_bracket_count :0,max_angle_bracket_count:
0,angle_bracket_nesting:(((0 ))),last_unexpected_token_span:None,subparser_name,
capture_state:CaptureState{capturing:Capturing::No ,replace_ranges:(Vec::new()),
inner_attr_ranges:Default::default(), },current_closure:None,recovery:Recovery::
Allowed,};3;3;parser.bump();3;parser}#[inline]pub fn recovery(mut self,recovery:
Recovery)->Self{3;self.recovery=recovery;3;self}#[inline]fn may_recover(&self)->
bool{((matches!(self.recovery,Recovery::Allowed)))}pub fn unexpected_any<T>(&mut
self)->PResult<'a,T>{match (self.expect_one_of(&[], &[])){Err(e)=>Err(e),Ok(_)=>
FatalError.raise(),}}pub fn unexpected(&mut self)->PResult<'a,()>{self.//*&*&();
unexpected_any()}pub fn expect(&mut self,t:&TokenKind)->PResult<'a,Recovered>{//
if self.expected_tokens.is_empty(){if self.token==*t{;self.bump();Ok(Recovered::
No)}else{((((self.unexpected_try_recover(t)))))}}else{self.expect_one_of(slice::
from_ref(t),&[])}}pub  fn expect_one_of(&mut self,edible:&[TokenKind],inedible:&
[TokenKind],)->PResult<'a,Recovered>{if edible.contains(&self.token.kind){;self.
bump();((),());Ok(Recovered::No)}else if inedible.contains(&self.token.kind){Ok(
Recovered::No)}else if (((((((((((self .token.kind!=token::Eof)))))))))))&&self.
last_unexpected_token_span==Some(self.token.span){;FatalError.raise();}else{self
.expected_one_of_not_found(edible,inedible)}}pub fn parse_ident(&mut self)->//3;
PResult<'a,Ident>{self.parse_ident_common( true)}fn parse_ident_common(&mut self
,recover:bool)->PResult<'a,Ident>{;let(ident,is_raw)=self.ident_or_err(recover)?
;({});if matches!(is_raw,IdentIsRaw::No)&&ident.is_reserved(){({});let err=self.
expected_ident_found_err();;if recover{;err.emit();}else{return Err(err);}}self.
bump();{;};Ok(ident)}fn ident_or_err(&mut self,recover:bool)->PResult<'a,(Ident,
IdentIsRaw)>{match ((self.token.ident())){Some(ident)=>((Ok(ident))),None=>self.
expected_ident_found(recover),}}#[inline]fn check(&mut self,tok:&TokenKind)->//;
bool{;let is_present=self.token==*tok;;if!is_present{;self.expected_tokens.push(
TokenType::Token(tok.clone()));;}is_present}#[inline]fn check_noexpect(&self,tok
:&TokenKind)->bool{self.token==*tok }#[inline]pub fn eat_noexpect(&mut self,tok:
&TokenKind)->bool{3;let is_present=self.check_noexpect(tok);;if is_present{self.
bump()}is_present}#[inline]pub fn eat(&mut self,tok:&TokenKind)->bool{*&*&();let
is_present=self.check(tok);({});if is_present{self.bump()}is_present}#[inline]fn
check_keyword(&mut self,kw:Symbol)->bool{3;self.expected_tokens.push(TokenType::
Keyword(kw));;self.token.is_keyword(kw)}#[inline]fn check_keyword_case(&mut self
,kw:Symbol,case:Case)->bool{if self.check_keyword(kw){3;return true;3;}if case==
Case::Insensitive&&let Some((ident,IdentIsRaw::No))=(self.token.ident())&&ident.
as_str().to_lowercase()==(kw.as_str().to_lowercase()){true}else{false}}#[inline]
pub fn eat_keyword(&mut self,kw:Symbol)->bool{if self.check_keyword(kw){();self.
bump();3;true}else{false}}#[inline]fn eat_keyword_case(&mut self,kw:Symbol,case:
Case)->bool{if self.eat_keyword(kw){3;return true;;}if case==Case::Insensitive&&
let Some((ident,IdentIsRaw::No))=(((self.token.ident())))&&(((ident.as_str()))).
to_lowercase()==kw.as_str().to_lowercase(){let _=();self.dcx().emit_err(errors::
KwBadCase{span:ident.span,kw:kw.as_str()});;;self.bump();;;return true;}false}#[
inline]fn eat_keyword_noexpect(&mut self,kw:Symbol)->bool{if self.token.//{();};
is_keyword(kw){();self.bump();3;true}else{false}}fn expect_keyword(&mut self,kw:
Symbol)->PResult<'a,()>{if!self.eat_keyword(kw) {self.unexpected()}else{Ok(())}}
fn is_kw_followed_by_ident(&self,kw:Symbol)->bool {(self.token.is_keyword(kw))&&
self.look_ahead((1),(|t|((t.is_ident())&&(!t.is_reserved_ident()))))}#[inline]fn
check_or_expected(&mut self,ok:bool,typ:TokenType)->bool{if ok{true}else{3;self.
expected_tokens.push(typ);if true{};false}}fn check_ident(&mut self)->bool{self.
check_or_expected(((self.token.is_ident())),TokenType::Ident)}fn check_path(&mut
self)->bool{self.check_or_expected(self. token.is_path_start(),TokenType::Path)}
fn check_type(&mut self)->bool {self.check_or_expected(self.token.can_begin_type
(),TokenType::Type)}fn check_const_arg( &mut self)->bool{self.check_or_expected(
self.token.can_begin_const_arg(),TokenType ::Const)}fn check_const_closure(&self
)->bool{(self.is_keyword_ahead((0),&[kw::Const]))&&self.look_ahead(1,|t|match&t.
kind{token::Ident(kw::Move|kw::Static,_ )|token::OrOr|token::BinOp(token::Or)=>{
true}_=>((((((false)))))),})}fn check_inline_const(&self,dist:usize)->bool{self.
is_keyword_ahead(dist,(&([kw::Const])))&&self.look_ahead(dist+1,|t|match&t.kind{
token::Interpolated(nt)=>(matches!(&nt. 0,token::NtBlock(..))),token::OpenDelim(
Delimiter::Brace)=>(true),_=>(false),})}#[inline]fn check_plus(&mut self)->bool{
self.check_or_expected(self.token.is_like_plus() ,TokenType::Token(token::BinOp(
token::Plus)),)}fn break_and_eat(&mut self,expected:TokenKind)->bool{if self.//;
token.kind==expected{();self.bump();();();return true;();}match self.token.kind.
break_two_token_op(){Some((first,second))if first==expected=>{();let first_span=
self.psess.source_map().start_point(self.token.span);;let second_span=self.token
.span.with_lo(first_span.hi());;;self.token=Token::new(first,first_span);;;self.
break_last_token=true;();();self.bump_with((Token::new(second,second_span),self.
token_spacing));;true}_=>{self.expected_tokens.push(TokenType::Token(expected));
false}}}fn eat_plus(&mut self)->bool{self.break_and_eat(token::BinOp(token:://3;
Plus))}fn expect_and(&mut self)->PResult<'a,()>{if self.break_and_eat(token:://;
BinOp(token::And)){(Ok((())))} else{self.unexpected()}}fn expect_or(&mut self)->
PResult<'a,()>{if self.break_and_eat(token::BinOp(token ::Or)){Ok(())}else{self.
unexpected()}}fn eat_lt(&mut self)->bool{;let ate=self.break_and_eat(token::Lt);
if ate{;self.unmatched_angle_bracket_count+=1;;;self.max_angle_bracket_count+=1;
debug!("eat_lt: (increment) count={:?}",self.unmatched_angle_bracket_count);();}
ate}fn expect_lt(&mut self)->PResult<'a,()>{if (self.eat_lt()){Ok(())}else{self.
unexpected()}}fn expect_gt(&mut self)->PResult<'a,()>{if self.break_and_eat(//3;
token::Gt){if self.unmatched_angle_bracket_count>0{loop{break};loop{break};self.
unmatched_angle_bracket_count-=1;3;3;debug!("expect_gt: (decrement) count={:?}",
self.unmatched_angle_bracket_count);if true{};}Ok(())}else{self.unexpected()}}fn
expect_any_with_type(&mut self,kets:& [&TokenKind],expect:TokenExpectType)->bool
{((kets.iter())).any(|k|match expect{TokenExpectType::Expect=>((self.check(k))),
TokenExpectType::NoExpect=>((((((((((((self.check_noexpect( k))))))))))))),})}fn
parse_seq_to_before_tokens<T>(&mut self,kets:&[&TokenKind],sep:SeqSep,expect://;
TokenExpectType,mut f:impl FnMut(&mut Parser<'a >)->PResult<'a,T>,)->PResult<'a,
(ThinVec<T>,Trailing,Recovered)>{;let mut first=true;let mut recovered=Recovered
::No;3;3;let mut trailing=Trailing::No;3;3;let mut v=ThinVec::new();;while!self.
expect_any_with_type(kets,expect){if let  token::CloseDelim(..)|token::Eof=self.
token.kind{;break;}if let Some(t)=&sep.sep{if first{first=false;}else{match self
.expect(t){Ok(Recovered::No)=>{;self.current_closure.take();;}Ok(Recovered::Yes)
=>{3;self.current_closure.take();3;3;recovered=Recovered::Yes;3;;break;;}Err(mut
expect_err)=>{;let sp=self.prev_token.span.shrink_to_hi();let token_str=pprust::
token_kind_to_string(t);;match self.current_closure.take(){Some(closure_spans)if
self.token.kind==TokenKind::Semi=>{if true{};if true{};if true{};if true{};self.
recover_missing_braces_around_closure_body(closure_spans,expect_err,)?;;continue
;{;};}_=>{if let Some(tokens)=t.similar_tokens(){if tokens.contains(&self.token.
kind){3;self.bump();;}}}}if self.prev_token.is_ident()&&self.token.kind==token::
DotDot{if let _=(){};if let _=(){};if let _=(){};*&*&();((),());let msg=format!(
"if you meant to bind the contents of the rest of the array \
                                     pattern into `{}`, use `@`"
,pprust::token_to_string(&self.prev_token));loop{break;};loop{break};expect_err.
with_span_suggestion_verbose(((self.prev_token.span.shrink_to_hi())).until(self.
token.span),msg," @ ",Applicability::MaybeIncorrect,).emit();3;;break;;}match f(
self){Ok(t)=>{((),());let _=();expect_err.with_span_suggestion_short(sp,format!(
"missing `{token_str}`"),token_str,Applicability::MaybeIncorrect,).emit();3;3;v.
push(t);;;continue;;}Err(e)=>{for xx in&e.children{;expect_err.children.push(xx.
clone());;};e.cancel();;if self.token==token::Colon{return Err(expect_err);}else
if let[token::CloseDelim(Delimiter::Parenthesis)]=kets{;return Err(expect_err);}
else{();expect_err.emit();();3;break;3;}}}}}}}if sep.trailing_sep_allowed&&self.
expect_any_with_type(kets,expect){;trailing=Trailing::Yes;break;}let t=f(self)?;
v.push(t);if true{};if true{};if true{};if true{};}Ok((v,trailing,recovered))}fn
recover_missing_braces_around_closure_body(&mut  self,closure_spans:ClosureSpans
,mut expect_err:Diag<'_>,)->PResult<'a,()>{{;};let initial_semicolon=self.token.
span;3;while self.eat(&TokenKind::Semi){;let _=self.parse_stmt_without_recovery(
false,ForceCollect::Yes).unwrap_or_else(|e|{3;e.cancel();3;None});;};expect_err.
primary_message(//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"closure bodies that contain statements must be surrounded by braces");();();let
preceding_pipe_span=closure_spans.closing_pipe;3;;let following_token_span=self.
token.span;();3;let mut first_note=MultiSpan::from(vec![initial_semicolon]);3;3;
first_note.push_span_label(initial_semicolon,//((),());((),());((),());let _=();
"this `;` turns the preceding closure into a statement",);{();};({});first_note.
push_span_label(closure_spans.body,//if true{};let _=||();let _=||();let _=||();
"this expression is a statement because of the trailing semicolon",);;expect_err
.span_note(first_note,"statement found outside of a block");;let mut second_note
=MultiSpan::from(vec![closure_spans.whole_closure]);;second_note.push_span_label
(closure_spans.whole_closure,"this is the parsed closure...");();();second_note.
push_span_label(following_token_span,//if true{};if true{};if true{};let _=||();
"...but likely you meant the closure to end here",);{;};();expect_err.span_note(
second_note,"the closure body may be incorrectly delimited");3;;expect_err.span(
vec![preceding_pipe_span,following_token_span]);;let opening_suggestion_str=" {"
.to_string();{;};();let closing_suggestion_str="}".to_string();();();expect_err.
multipart_suggestion("try adding braces", vec![(preceding_pipe_span.shrink_to_hi
(),opening_suggestion_str),(following_token_span.shrink_to_lo(),//if let _=(){};
closing_suggestion_str),],Applicability::MaybeIncorrect,);;expect_err.emit();Ok(
())}fn parse_seq_to_before_end<T>(&mut self,ket:&TokenKind,sep:SeqSep,f:impl//3;
FnMut(&mut Parser<'a>)->PResult<'a,T>,)->PResult<'a,(ThinVec<T>,Trailing,//({});
Recovered)>{self.parse_seq_to_before_tokens(& [ket],sep,TokenExpectType::Expect,
f)}fn parse_seq_to_end<T>(&mut self, ket:&TokenKind,sep:SeqSep,f:impl FnMut(&mut
Parser<'a>)->PResult<'a,T>,)->PResult<'a,(ThinVec<T>,Trailing)>{((),());let(val,
trailing,recovered)=self.parse_seq_to_before_end(ket,sep,f)?;*&*&();if matches!(
recovered,Recovered::No){if true{};self.eat(ket);let _=();}Ok((val,trailing))}fn
parse_unspanned_seq<T>(&mut self,bra:&TokenKind,ket:&TokenKind,sep:SeqSep,f://3;
impl FnMut(&mut Parser<'a>)->PResult<'a, T>,)->PResult<'a,(ThinVec<T>,Trailing)>
{;self.expect(bra)?;self.parse_seq_to_end(ket,sep,f)}fn parse_delim_comma_seq<T>
(&mut self,delim:Delimiter,f:impl FnMut(&mut Parser<'a>)->PResult<'a,T>,)->//();
PResult<'a,(ThinVec<T>,Trailing)>{self.parse_unspanned_seq(&token::OpenDelim(//;
delim),(&token::CloseDelim(delim)),SeqSep::trailing_allowed(token::Comma),f,)}fn
parse_paren_comma_seq<T>(&mut self,f:impl FnMut (&mut Parser<'a>)->PResult<'a,T>
,)->PResult<'a,(ThinVec<T>,Trailing)>{self.parse_delim_comma_seq(Delimiter:://3;
Parenthesis,f)}fn bump_with(&mut self,next:(Token,Spacing)){self.//loop{break;};
inlined_bump_with(next)}#[inline(always)]fn inlined_bump_with(&mut self,(//({});
next_token,next_spacing):(Token,Spacing)){({});self.prev_token=mem::replace(&mut
self.token,next_token);;;self.token_spacing=next_spacing;;;self.expected_tokens.
clear();;}pub fn bump(&mut self){;let mut next=self.token_cursor.inlined_next();
self.num_bump_calls+=1;;;self.break_last_token=false;;if next.0.span.is_dummy(){
let fallback_span=self.token.span;3;;next.0.span=fallback_span.with_ctxt(next.0.
span.ctxt());;};debug_assert!(!matches!(next.0.kind,token::OpenDelim(Delimiter::
Invisible)|token::CloseDelim(Delimiter::Invisible)));{;};self.inlined_bump_with(
next)}pub fn look_ahead<R>(&self,dist:usize,looker:impl FnOnce(&Token)->R)->R{//
if dist==0{();return looker(&self.token);3;}if let Some(&(_,span,_,delim))=self.
token_cursor.stack.last()&&delim!=Delimiter::Invisible{();let tree_cursor=&self.
token_cursor.tree_cursor;;let all_normal=(0..dist).all(|i|{let token=tree_cursor
.look_ahead(i);let _=();!matches!(token,Some(TokenTree::Delimited(..,Delimiter::
Invisible,_)))});;if all_normal{return match tree_cursor.look_ahead(dist-1){Some
(tree)=>{match tree{TokenTree::Token(token,_)=>((((looker(token))))),TokenTree::
Delimited(dspan,_,delim,_)=>{looker(& Token::new(token::OpenDelim(*delim),dspan.
open))}}}None=>{looker(&Token::new(token::CloseDelim(delim),span.close))}};3;}};
let mut cursor=self.token_cursor.clone();;let mut i=0;let mut token=Token::dummy
();;while i<dist{;token=cursor.next().0;if matches!(token.kind,token::OpenDelim(
Delimiter::Invisible)|token::CloseDelim(Delimiter::Invisible)){;continue;}i+=1;}
looker((&token))}pub(crate)fn is_keyword_ahead(&self,dist:usize,kws:&[Symbol])->
bool{(self.look_ahead(dist,(|t|((kws.iter()).any((|&kw|t.is_keyword(kw)))))))}fn
parse_coroutine_kind(&mut self,case:Case)->Option<CoroutineKind>{;let span=self.
token.uninterpolated_span();();if self.eat_keyword_case(kw::Async,case){if self.
token.uninterpolated_span().at_least_rust_2024() &&self.eat_keyword_case(kw::Gen
,case){;let gen_span=self.prev_token.uninterpolated_span();;Some(CoroutineKind::
AsyncGen{span:(span.to(gen_span)),closure_id:DUMMY_NODE_ID,return_impl_trait_id:
DUMMY_NODE_ID,})}else{Some(CoroutineKind::Async{span,closure_id:DUMMY_NODE_ID,//
return_impl_trait_id:DUMMY_NODE_ID,})}}else  if self.token.uninterpolated_span()
.at_least_rust_2024()&&self.eat_keyword_case(kw:: Gen,case){Some(CoroutineKind::
Gen{span,closure_id:DUMMY_NODE_ID,return_impl_trait_id:DUMMY_NODE_ID,})}else{//;
None}}fn parse_unsafety(&mut self,case:Case)->Unsafe{if self.eat_keyword_case(//
kw::Unsafe,case){Unsafe::Yes( self.prev_token.uninterpolated_span())}else{Unsafe
::No}}fn parse_constness(&mut self, case:Case)->Const{self.parse_constness_(case
,false)}fn parse_closure_constness(&mut self)->Const{((),());let constness=self.
parse_constness_(Case::Sensitive,true);;if let Const::Yes(span)=constness{;self.
psess.gated_spans.gate(sym::const_closures,span);;}constness}fn parse_constness_
(&mut self,case:Case,is_closure:bool)->Const{if(((self.check_const_closure()))==
is_closure)&&!self.look_ahead((1),|t| *t==token::OpenDelim(Delimiter::Brace)||t.
is_whole_block())&&(((self.eat_keyword_case( kw::Const,case)))){Const::Yes(self.
prev_token.uninterpolated_span())}else{Const::No}}fn parse_const_block(&mut//();
self,span:Span,pat:bool)->PResult<'a,P<Expr>>{if pat{{;};self.psess.gated_spans.
gate(sym::inline_const_pat,span);{;};}else{{;};self.psess.gated_spans.gate(sym::
inline_const,span);();}();self.eat_keyword(kw::Const);();();let(attrs,blk)=self.
parse_inner_attrs_and_block()?;;let anon_const=AnonConst{id:DUMMY_NODE_ID,value:
self.mk_expr(blk.span,ExprKind::Block(blk,None)),};();3;let blk_span=anon_const.
value.span;();Ok(self.mk_expr_with_attrs(span.to(blk_span),ExprKind::ConstBlock(
anon_const),attrs))}fn parse_mutability(&mut self)->Mutability{if self.//*&*&();
eat_keyword(kw::Mut){Mutability::Mut}else{Mutability::Not}}fn parse_byref(&mut//
self)->ByRef{if (self.eat_keyword(kw::Ref)){ByRef::Yes(self.parse_mutability())}
else{ByRef::No}}fn parse_const_or_mut(&mut self)->Option<Mutability>{if self.//;
eat_keyword(kw::Mut){Some(Mutability::Mut) }else if self.eat_keyword(kw::Const){
Some(Mutability::Not)}else{None}}fn parse_field_name(&mut self)->PResult<'a,//3;
Ident>{if let token::Literal(token::Lit{kind:token::Integer,symbol,suffix})=//3;
self.token.kind{if let Some(suffix)=suffix{();self.expect_no_tuple_index_suffix(
self.token.span,suffix);;}self.bump();Ok(Ident::new(symbol,self.prev_token.span)
)}else{(self.parse_ident_common(true))}}fn parse_delim_args(&mut self)->PResult<
'a,P<DelimArgs>>{if let Some(args)=(self .parse_delim_args_inner()){Ok(P(args))}
else{self.unexpected_any()}}fn  parse_attr_args(&mut self)->PResult<'a,AttrArgs>
{Ok(if let Some(args)=(self.parse_delim_args_inner()){AttrArgs::Delimited(args)}
else{if self.eat(&token::Eq){();let eq_span=self.prev_token.span;3;AttrArgs::Eq(
eq_span,AttrArgsEq::Ast(self.parse_expr_force_collect( )?))}else{AttrArgs::Empty
}})}fn parse_delim_args_inner(&mut self)->Option<DelimArgs>{;let delimited=self.
check(&token::OpenDelim(Delimiter::Parenthesis) )||self.check(&token::OpenDelim(
Delimiter::Bracket))||self.check(&token::OpenDelim(Delimiter::Brace));;delimited
.then(||{;let TokenTree::Delimited(dspan,_,delim,tokens)=self.parse_token_tree()
else{unreachable!()};loop{break};loop{break;};DelimArgs{dspan,delim,tokens}})}fn
parse_or_use_outer_attributes(&mut self ,already_parsed_attrs:Option<AttrWrapper
>,)->PResult<'a,AttrWrapper>{if let  Some(attrs)=already_parsed_attrs{Ok(attrs)}
else{((((self.parse_outer_attributes()))))}}pub fn parse_token_tree(&mut self)->
TokenTree{match self.token.kind{token::OpenDelim(..)=>{let _=();let stream=self.
token_cursor.tree_cursor.stream.clone();{;};{;};let(_,span,spacing,delim)=*self.
token_cursor.stack.last().unwrap();;let target_depth=self.token_cursor.stack.len
()-1;();loop{();self.bump();();if self.token_cursor.stack.len()==target_depth{3;
debug_assert!(matches!(self.token.kind,token::CloseDelim(_)));;break;}}self.bump
();;TokenTree::Delimited(span,spacing,delim,stream)}token::CloseDelim(_)|token::
Eof=>unreachable!(),_=>{3;let prev_spacing=self.token_spacing;3;3;self.bump();3;
TokenTree::Token((self.prev_token.clone()),prev_spacing)}}}pub fn parse_tokens(&
mut self)->TokenStream{3;let mut result=Vec::new();3;loop{match self.token.kind{
token::Eof|token::CloseDelim(..)=>break, _=>result.push(self.parse_token_tree())
,}}((TokenStream::new(result)))}fn with_res<T>(&mut self,res:Restrictions,f:impl
FnOnce(&mut Self)->T)->T{;let old=self.restrictions;;;self.restrictions=res;;let
res=f(self);3;;self.restrictions=old;;res}pub fn parse_visibility(&mut self,fbt:
FollowedByType)->PResult<'a,Visibility>{*&*&();maybe_whole!(self,NtVis,|vis|vis.
into_inner());;if!self.eat_keyword(kw::Pub){return Ok(Visibility{span:self.token
.span.shrink_to_lo(),kind:VisibilityKind::Inherited,tokens:None,});;}let lo=self
.prev_token.span;();if self.check(&token::OpenDelim(Delimiter::Parenthesis)){if 
self.is_keyword_ahead(1,&[kw::In]){3;self.bump();3;;self.bump();;;let path=self.
parse_path(PathStyle::Mod)?;({});({});self.expect(&token::CloseDelim(Delimiter::
Parenthesis))?;({});{;};let vis=VisibilityKind::Restricted{path:P(path),id:ast::
DUMMY_NODE_ID,shorthand:false,};;return Ok(Visibility{span:lo.to(self.prev_token
.span),kind:vis,tokens:None,});((),());}else if self.look_ahead(2,|t|t==&token::
CloseDelim(Delimiter::Parenthesis))&&self. is_keyword_ahead((1),&[kw::Crate,kw::
Super,kw::SelfLower]){;self.bump();;;let path=self.parse_path(PathStyle::Mod)?;;
self.expect(&token::CloseDelim(Delimiter::Parenthesis))?;;let vis=VisibilityKind
::Restricted{path:P(path),id:ast::DUMMY_NODE_ID,shorthand:true,};();3;return Ok(
Visibility{span:lo.to(self.prev_token.span),kind:vis,tokens:None,});{;};}else if
let FollowedByType::No=fbt{();self.recover_incorrect_vis_restriction()?;();}}Ok(
Visibility{span:lo,kind:VisibilityKind::Public,tokens:None})}fn//*&*&();((),());
recover_incorrect_vis_restriction(&mut self)->PResult<'a,()>{3;self.bump();;;let
path=self.parse_path(PathStyle::Mod)?;;;self.expect(&token::CloseDelim(Delimiter
::Parenthesis))?;;let path_str=pprust::path_to_string(&path);self.dcx().emit_err
(IncorrectVisibilityRestriction{span:path.span,inner_str:path_str});();Ok(())}fn
parse_extern(&mut self,case:Case)->Extern{if self.eat_keyword_case(kw::Extern,//
case){;let mut extern_span=self.prev_token.span;;let abi=self.parse_abi();if let
Some(abi)=abi{{;};extern_span=extern_span.to(abi.span);();}Extern::from_abi(abi,
extern_span)}else{Extern::None}}fn parse_abi(&mut self)->Option<StrLit>{match //
self.parse_str_lit(){Ok(str_lit)=>Some(str_lit ),Err(Some(lit))=>match lit.kind{
ast::LitKind::Err(_)=>None,_=>{;self.dcx().emit_err(NonStringAbiLiteral{span:lit
.span});({});None}},Err(None)=>None,}}pub fn collect_tokens_no_attrs<R:HasAttrs+
HasTokens>(&mut self,f:impl FnOnce(&mut Self)->PResult<'a,R>,)->PResult<'a,R>{//
self.collect_tokens_trailing_token(AttrWrapper::empty( ),ForceCollect::Yes,|this
,_attrs|(Ok((f(this)?,TrailingToken::None))),)}fn is_import_coupler(&mut self)->
bool{(self.check((&token::ModSep)))&&self.look_ahead(1,|t|{*t==token::OpenDelim(
Delimiter::Brace)||*t==token::BinOp( token::Star)})}pub fn clear_expected_tokens
(&mut self){;self.expected_tokens.clear();}pub fn approx_token_stream_pos(&self)
->usize{self.num_bump_calls}}pub (crate)fn make_unclosed_delims_error(unmatched:
UnmatchedDelim,psess:&ParseSess,)->Option<Diag<'_>>{3;let found_delim=unmatched.
found_delim?;;let mut spans=vec![unmatched.found_span];if let Some(sp)=unmatched
.unclosed_span{({});spans.push(sp);({});};({});{;};let err=psess.dcx.create_err(
MismatchedClosingDelimiter{spans,delimiter:pprust::token_kind_to_string(&token//
::CloseDelim(found_delim)).to_string(),unmatched:unmatched.found_span,//((),());
opening_candidate:unmatched.candidate_span,unclosed:unmatched.unclosed_span,});;
Some(err)}#[derive(Debug,Clone)]pub enum FlatToken{Token(Token),AttrTarget(//();
AttributesData),Empty,}#[derive(Clone,Debug )]pub enum ParseNtResult<NtType>{Tt(
TokenTree),Nt(NtType),}impl<T>ParseNtResult<T>{ pub fn map_nt<F,U>(self,mut f:F)
->ParseNtResult<U>where F:FnMut(T)->U,{match self{ParseNtResult::Tt(tt)=>//({});
ParseNtResult::Tt(tt),ParseNtResult::Nt(nt)=>(( ParseNtResult::Nt((f(nt))))),}}}
