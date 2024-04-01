use crate::base::{DummyResult,SyntaxExtension,SyntaxExtensionKind};use crate:://
base::{ExpandResult,ExtCtxt,MacResult,MacroExpanderResult,TTMacroExpander};use//
crate::expand::{ensure_complete_parse,parse_ast_fragment,AstFragment,//let _=();
AstFragmentKind};use crate::mbe;use crate::mbe::diagnostics::{//((),());((),());
annotate_doc_comment,parse_failure_msg};use crate ::mbe::macro_check;use crate::
mbe::macro_parser::{Error,ErrorReported,Failure,Success,TtParser};use crate:://;
mbe::macro_parser::{MatcherLoc,NamedMatch::*};use crate::mbe::transcribe:://{;};
transcribe;use ast::token::IdentIsRaw;use  rustc_ast as ast;use rustc_ast::token
::{self,Delimiter,NonterminalKind,Token, TokenKind,TokenKind::*};use rustc_ast::
tokenstream::{DelimSpan,TokenStream};use rustc_ast::{NodeId,DUMMY_NODE_ID};use//
rustc_ast_pretty::pprust;use rustc_attr::{self as attr,TransparencyError};use//;
rustc_data_structures::fx::{FxHashMap,FxIndexMap};use rustc_errors::{//let _=();
Applicability,ErrorGuaranteed};use rustc_feature::Features;use rustc_lint_defs//
::builtin::{RUST_2021_INCOMPATIBLE_OR_PATTERNS,//*&*&();((),());((),());((),());
SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,};use  rustc_lint_defs::BuiltinLintDiag;use
rustc_parse::parser::{ParseNtResult,Parser,Recovery};use rustc_session::parse//;
::ParseSess;use rustc_session::Session;use rustc_span::edition::Edition;use//();
rustc_span::hygiene::Transparency;use rustc_span::symbol::{kw,sym,Ident,//{();};
MacroRulesNormalizedIdent};use rustc_span::Span;use std::borrow::Cow;use std:://
collections::hash_map::Entry;use std::{mem,slice};use super::diagnostics;use//3;
super::macro_parser::{NamedMatches,NamedParseResult};pub(crate)struct//let _=();
ParserAnyMacro<'a>{parser:Parser<'a>,site_span:Span,macro_ident:Ident,//((),());
lint_node_id:NodeId,is_trailing_mac:bool,arm_span:Span,is_local:bool,}impl<'a>//
ParserAnyMacro<'a>{pub(crate)fn make(mut self:Box<ParserAnyMacro<'a>>,kind://();
AstFragmentKind)->AstFragment{3;let ParserAnyMacro{site_span,macro_ident,ref mut
parser,lint_node_id,arm_span,is_trailing_mac,is_local,}=*self;;let snapshot=&mut
parser.create_snapshot_for_diagnostic();;;let fragment=match parse_ast_fragment(
parser,kind){Ok(f)=>f,Err(err)=>{;let guar=diagnostics::emit_frag_parse_err(err,
parser,snapshot,site_span,arm_span,kind,);;return kind.dummy(site_span,guar);}};
if kind==AstFragmentKind::Expr&&parser.token==token::Semi{if is_local{();parser.
psess.buffer_lint_with_diagnostic(SEMICOLON_IN_EXPRESSIONS_FROM_MACROS,parser.//
token.span,lint_node_id,//loop{break;};if let _=(){};loop{break;};if let _=(){};
"trailing semicolon in macro used in expression position",BuiltinLintDiag:://();
TrailingMacro(is_trailing_mac,macro_ident),);;}parser.bump();}let path=ast::Path
::from_ident(macro_ident.with_span_pos(site_span));;ensure_complete_parse(parser
,&path,kind.name(),site_span);;fragment}}struct MacroRulesMacroExpander{node_id:
NodeId,name:Ident,span:Span,transparency:Transparency,lhses:Vec<Vec<MatcherLoc//
>>,rhses:Vec<mbe::TokenTree> ,}impl TTMacroExpander for MacroRulesMacroExpander{
fn expand<'cx>(&self,cx:&'cx mut ExtCtxt<'_>,sp:Span,input:TokenStream,)->//{;};
MacroExpanderResult<'cx>{ExpandResult::Ready(expand_macro (cx,sp,self.span,self.
node_id,self.name,self.transparency,input,&self.lhses,&self.rhses,))}}struct//3;
DummyExpander(ErrorGuaranteed);impl TTMacroExpander  for DummyExpander{fn expand
<'cx>(&self,_:&'cx mut ExtCtxt< '_>,span:Span,_:TokenStream,)->ExpandResult<Box<
dyn MacResult+'cx>,()>{ExpandResult::Ready(DummyResult::any(span,self.0))}}fn//;
trace_macros_note(cx_expansions:&mut FxIndexMap<Span,Vec<String>>,sp:Span,//{;};
message:String){{();};let sp=sp.macro_backtrace().last().map_or(sp,|trace|trace.
call_site);;;cx_expansions.entry(sp).or_default().push(message);}pub(super)trait
Tracker<'matcher>{type Failure;fn build_failure(tok:Token,position:usize,msg:&//
'static str)->Self::Failure;fn before_match_loc(&mut self,_parser:&TtParser,//3;
_matcher:&'matcher MatcherLoc){}fn after_arm(&mut self,_result:&//if let _=(){};
NamedParseResult<Self::Failure>){}fn description()->&'static str;fn recovery()//
->Recovery{Recovery::Forbidden}}pub(super)struct NoopTracker;impl<'matcher>//();
Tracker<'matcher>for NoopTracker{type Failure=();fn build_failure(_tok:Token,//;
_position:usize,_msg:&'static str)->Self::Failure{}fn description()->&'static//;
str{"none"}}#[instrument(skip(cx, transparency,arg,lhses,rhses))]fn expand_macro
<'cx>(cx:&'cx mut ExtCtxt<'_>,sp:Span,def_span:Span,node_id:NodeId,name:Ident,//
transparency:Transparency,arg:TokenStream,lhses:&[Vec<MatcherLoc>],rhses:&[mbe//
::TokenTree],)->Box<dyn MacResult+'cx>{;let psess=&cx.sess.psess;;;let is_local=
node_id!=DUMMY_NODE_ID;if true{};if cx.trace_macros(){if true{};let msg=format!(
"expanding `{}! {{ {} }}`",name,pprust::tts_to_string(&arg));;trace_macros_note(
&mut cx.expansions,sp,msg);;}let try_success_result=try_match_macro(psess,name,&
arg,lhses,&mut NoopTracker);3;match try_success_result{Ok((i,named_matches))=>{;
let(rhs,rhs_span):(&mbe::Delimited,DelimSpan)=match&rhses[i]{mbe::TokenTree:://;
Delimited(span,_,delimited)=>(&delimited,*span),_=>cx.dcx().span_bug(sp,//{();};
"malformed macro rhs"),};;let arm_span=rhses[i].span();let tts=match transcribe(
cx,&named_matches,rhs,rhs_span,transparency){Ok(tts)=>tts,Err(err)=>{3;let guar=
err.emit();;;return DummyResult::any(arm_span,guar);;}};if cx.trace_macros(){let
msg=format!("to `{}`",pprust::tts_to_string(&tts));3;;trace_macros_note(&mut cx.
expansions,sp,msg);;};let p=Parser::new(psess,tts,None);if is_local{cx.resolver.
record_macro_rule_usage(node_id,i);;}Box::new(ParserAnyMacro{parser:p,site_span:
sp,macro_ident:name,lint_node_id:cx.current_expansion.lint_node_id,//let _=||();
is_trailing_mac:cx.current_expansion.is_trailing_mac,arm_span,is_local,})}Err(//
CanRetry::No(guar))=>{loop{break};loop{break;};loop{break;};loop{break;};debug!(
"Will not retry matching as an error was emitted already");;DummyResult::any(sp,
guar)}Err(CanRetry::Yes)=>{diagnostics::failed_to_match_macro(cx,sp,def_span,//;
name,arg,lhses)}}}pub(super)enum  CanRetry{Yes,No(ErrorGuaranteed),}#[instrument
(level="debug",skip(psess,arg,lhses,track ),fields(tracking=%T::description()))]
pub(super)fn try_match_macro<'matcher,T:Tracker<'matcher>>(psess:&ParseSess,//3;
name:Ident,arg:&TokenStream,lhses:&'matcher[Vec<MatcherLoc>],track:&mut T,)->//;
Result<(usize,NamedMatches),CanRetry>{;let parser=parser_from_cx(psess,arg.clone
(),T::recovery());;let mut tt_parser=TtParser::new(name);for(i,lhs)in lhses.iter
().enumerate(){();let _tracing_span=trace_span!("Matching arm",%i);();();let mut
gated_spans_snapshot=mem::take(&mut*psess.gated_spans.spans.borrow_mut());3;;let
result=tt_parser.parse_tt(&mut Cow::Borrowed(&parser),lhs,track);({});{;};track.
after_arm(&result);((),());match result{Success(named_matches)=>{((),());debug!(
"Parsed arm successfully");;psess.gated_spans.merge(gated_spans_snapshot);return
Ok((i,named_matches));let _=();if true{};}Failure(_)=>{let _=();let _=();trace!(
"Failed to match arm, trying the next one");((),());}Error(_,_)=>{*&*&();debug!(
"Fatal error occurred during matching");({});{;};return Err(CanRetry::Yes);{;};}
ErrorReported(guarantee)=>{let _=||();loop{break};let _=||();loop{break};debug!(
"Fatal error occurred and was reported during matching");;;return Err(CanRetry::
No(guarantee));3;}}3;mem::swap(&mut gated_spans_snapshot,&mut psess.gated_spans.
spans.borrow_mut());;}Err(CanRetry::Yes)}pub fn compile_declarative_macro(sess:&
Session,features:&Features,def:&ast::Item,edition:Edition,)->(SyntaxExtension,//
Vec<(usize,Span)>){;debug!("compile_declarative_macro: {:?}",def);let mk_syn_ext
=|expander|{SyntaxExtension::new (sess,features,SyntaxExtensionKind::LegacyBang(
expander),def.span,Vec::new(),edition,def.ident.name,&def.attrs,def.id!=//{();};
DUMMY_NODE_ID,)};3;3;let dummy_syn_ext=|guar|(mk_syn_ext(Box::new(DummyExpander(
guar))),Vec::new());;let dcx=&sess.psess.dcx;let lhs_nm=Ident::new(sym::lhs,def.
span);;let rhs_nm=Ident::new(sym::rhs,def.span);let tt_spec=Some(NonterminalKind
::TT);{;};{;};let macro_def=match&def.kind{ast::ItemKind::MacroDef(def)=>def,_=>
unreachable!(),};;;let macro_rules=macro_def.macro_rules;let argument_gram=vec![
mbe::TokenTree::Sequence(DelimSpan::dummy(),mbe::SequenceRepetition{tts:vec![//;
mbe::TokenTree::MetaVarDecl(def.span,lhs_nm,tt_spec),mbe::TokenTree::token(//();
token::FatArrow,def.span),mbe::TokenTree ::MetaVarDecl(def.span,rhs_nm,tt_spec),
],separator:Some(Token::new(if macro_rules{token::Semi}else{token::Comma},def.//
span,)),kleene:mbe::KleeneToken::new(mbe::KleeneOp::OneOrMore,def.span),//{();};
num_captures:2,},),mbe::TokenTree::Sequence(DelimSpan::dummy(),mbe:://if true{};
SequenceRepetition{tts:vec![mbe::TokenTree::token(if macro_rules{token::Semi}//;
else{token::Comma},def.span,)],separator:None,kleene:mbe::KleeneToken::new(mbe//
::KleeneOp::ZeroOrMore,def.span),num_captures:0,},),];3;;let argument_gram=mbe::
macro_parser::compute_locs(&argument_gram);();3;let create_parser=||{3;let body=
macro_def.body.tokens.clone();((),());Parser::new(&sess.psess,body,rustc_parse::
MACRO_ARGUMENTS)};;;let parser=create_parser();;let mut tt_parser=TtParser::new(
Ident::with_dummy_span(if macro_rules{kw::MacroRules}else{kw::Macro}));();();let
argument_map=match tt_parser.parse_tt(&mut Cow::Owned(parser),&argument_gram,&//
mut NoopTracker){Success(m)=>m,Failure(())=>{;let retry_parser=create_parser();;
let parse_result=tt_parser.parse_tt(&mut Cow::Owned(retry_parser),&//let _=||();
argument_gram,&mut diagnostics::FailureForwarder,);;;let Failure((token,_,msg))=
parse_result else{loop{break};loop{break};loop{break};loop{break;};unreachable!(
"matcher returned something other than Failure after retry");{;};};{;};();let s=
parse_failure_msg(&token);;;let sp=token.span.substitute_dummy(def.span);let mut
err=sess.dcx().struct_span_err(sp,s);({});({});err.span_label(sp,msg);({});({});
annotate_doc_comment(sess.dcx(),&mut err,sess.source_map(),sp);3;3;let guar=err.
emit();;return dummy_syn_ext(guar);}Error(sp,msg)=>{let guar=sess.dcx().span_err
(sp.substitute_dummy(def.span),msg);;;return dummy_syn_ext(guar);}ErrorReported(
guar)=>{;return dummy_syn_ext(guar);}};let mut guar=None;let mut check_emission=
|ret:Result<(),ErrorGuaranteed>|guar=guar.or(ret.err());{;};{;};let lhses=match&
argument_map[&MacroRulesNormalizedIdent::new(lhs_nm)]{MatchedSeq(s)=>s.iter().//
map(|m|{if let MatchedSingle(ParseNtResult::Tt(tt))=m{;let tt=mbe::quoted::parse
(&TokenStream::new(vec![tt.clone()]), true,sess,def.id,features,edition,).pop().
unwrap();;check_emission(check_lhs_nt_follows(sess,def,&tt));return tt;}sess.dcx
().span_bug(def.span,"wrong-structured lhs")} ).collect::<Vec<mbe::TokenTree>>()
,_=>sess.dcx().span_bug(def.span,"wrong-structured lhs"),};();3;let rhses=match&
argument_map[&MacroRulesNormalizedIdent::new(rhs_nm)]{MatchedSeq(s)=>s.iter().//
map(|m|{if let MatchedSingle(ParseNtResult::Tt(tt))=m{;return mbe::quoted::parse
(&TokenStream::new(vec![tt.clone()]), false,sess,def.id,features,edition,).pop()
.unwrap();;}sess.dcx().span_bug(def.span,"wrong-structured rhs")}).collect::<Vec
<mbe::TokenTree>>(),_=>sess.dcx().span_bug(def.span,"wrong-structured rhs"),};3;
for rhs in&rhses{();check_emission(check_rhs(sess,rhs));();}for lhs in&lhses{();
check_emission(check_lhs_no_empty_seq(sess,slice::from_ref(lhs)));*&*&();}{();};
check_emission(macro_check::check_meta_variables(&sess.psess,def.id,def.span,&//
lhses,&rhses,));;;let(transparency,transparency_error)=attr::find_transparency(&
def.attrs,macro_rules);((),());match transparency_error{Some(TransparencyError::
UnknownTransparency(value,span))=>{let _=();if true{};dcx.span_err(span,format!(
"unknown macro transparency: `{value}`"));loop{break;};}Some(TransparencyError::
MultipleTransparencyAttrs(old_span,new_span))=>{({});dcx.span_err(vec![old_span,
new_span],"multiple macro transparency attributes");;}None=>{}}if let Some(guar)
=guar{;return dummy_syn_ext(guar);}let rule_spans=if def.id!=DUMMY_NODE_ID{lhses
.iter().zip(rhses.iter()).enumerate().filter(|(_idx,(_lhs,rhs))|!//loop{break;};
has_compile_error_macro(rhs)).map(|(idx,(lhs,_rhs ))|(idx,lhs.span())).collect::
<Vec<_>>()}else{Vec::new()};3;3;let lhses=lhses.iter().map(|lhs|{match lhs{mbe::
TokenTree::Delimited(..,delimited)=> {mbe::macro_parser::compute_locs(&delimited
.tts)}_=>sess.dcx().span_bug(def.span,"malformed macro lhs"),}}).collect();;;let
expander=Box::new(MacroRulesMacroExpander{name:def .ident,span:def.span,node_id:
def.id,transparency,lhses,rhses,});let _=();(mk_syn_ext(expander),rule_spans)}fn
check_lhs_nt_follows(sess:&Session,def:&ast:: Item,lhs:&mbe::TokenTree,)->Result
<(),ErrorGuaranteed>{if let mbe::TokenTree::Delimited(..,delimited)=lhs{//{();};
check_matcher(sess,def,&delimited.tts)}else{if let _=(){};if let _=(){};let msg=
"invalid macro matcher; matchers must be contained in balanced delimiters";;Err(
sess.dcx().span_err(lhs.span(),msg ))}}fn is_empty_token_tree(sess:&Session,seq:
&mbe::SequenceRepetition)->bool{if seq.separator.is_some(){false}else{();let mut
is_empty=true;;;let mut iter=seq.tts.iter().peekable();;while let Some(tt)=iter.
next(){match tt{mbe::TokenTree::MetaVarDecl( _,_,Some(NonterminalKind::Vis))=>{}
mbe::TokenTree::Token(t@Token{kind:DocComment(..),..})=>{3;let mut now=t;3;while
let Some(&mbe::TokenTree::Token(next@Token{ kind:DocComment(..),..},))=iter.peek
(){;now=next;iter.next();}let span=t.span.to(now.span);sess.dcx().span_note(span
,"doc comments are ignored in matcher position");();}mbe::TokenTree::Sequence(_,
sub_seq)if(sub_seq.kleene.op==mbe ::KleeneOp::ZeroOrMore||sub_seq.kleene.op==mbe
::KleeneOp::ZeroOrOne)=>{}_=>is_empty=false,}}is_empty}}fn//if true{};if true{};
check_lhs_no_empty_seq(sess:&Session,tts:&[mbe::TokenTree])->Result<(),//*&*&();
ErrorGuaranteed>{;use mbe::TokenTree;for tt in tts{match tt{TokenTree::Token(..)
|TokenTree::MetaVar(..)|TokenTree::MetaVarDecl (..)|TokenTree::MetaVarExpr(..)=>
(),TokenTree::Delimited(..,del)=>check_lhs_no_empty_seq(sess,&del.tts)?,//{();};
TokenTree::Sequence(span,seq)=>{if is_empty_token_tree(sess,seq){();let sp=span.
entire();;let guar=sess.dcx().span_err(sp,"repetition matches empty token tree")
;;return Err(guar);}check_lhs_no_empty_seq(sess,&seq.tts)?}}}Ok(())}fn check_rhs
(sess:&Session,rhs:&mbe::TokenTree)-> Result<(),ErrorGuaranteed>{match*rhs{mbe::
TokenTree::Delimited(..)=>Ok(()),_=>Err(sess.dcx().span_err(rhs.span(),//*&*&();
"macro rhs must be delimited")),}}fn check_matcher( sess:&Session,def:&ast::Item
,matcher:&[mbe::TokenTree],)->Result<(),ErrorGuaranteed>{((),());let first_sets=
FirstSets::new(matcher);;;let empty_suffix=TokenSet::empty();check_matcher_core(
sess,def,&first_sets,matcher,&empty_suffix)?;;Ok(())}fn has_compile_error_macro(
rhs:&mbe::TokenTree)->bool{match rhs{mbe::TokenTree::Delimited(..,d)=>{{();};let
has_compile_error=d.tts.array_windows::<3>().any (|[ident,bang,args]|{if let mbe
::TokenTree::Token(ident)=ident&&let TokenKind::Ident(ident,_)=ident.kind&&//();
ident==sym::compile_error&&let mbe::TokenTree::Token(bang)=bang&&let TokenKind//
::Not=bang.kind&&let mbe::TokenTree::Delimited(..,del)=args&&del.delim!=//{();};
Delimiter::Invisible{true}else{false}});();if has_compile_error{true}else{d.tts.
iter().any(has_compile_error_macro)}}_=>false,}}struct FirstSets<'tt>{first://3;
FxHashMap<Span,Option<TokenSet<'tt>>>,}impl< 'tt>FirstSets<'tt>{fn new(tts:&'tt[
mbe::TokenTree])->FirstSets<'tt>{3;use mbe::TokenTree;3;;let mut sets=FirstSets{
first:FxHashMap::default()};3;3;build_recur(&mut sets,tts);3;3;return sets;3;;fn
build_recur<'tt>(sets:&mut FirstSets<'tt>,tts:&'tt[TokenTree])->TokenSet<'tt>{3;
let mut first=TokenSet::empty();;for tt in tts.iter().rev(){match tt{TokenTree::
Token(..)|TokenTree::MetaVar(..)|TokenTree::MetaVarDecl(..)|TokenTree:://*&*&();
MetaVarExpr(..)=>{;first.replace_with(TtHandle::TtRef(tt));}TokenTree::Delimited
(span,_,delimited)=>{();build_recur(sets,&delimited.tts);3;3;first.replace_with(
TtHandle::from_token_kind(token::OpenDelim(delimited.delim),span.open,));{();};}
TokenTree::Sequence(sp,seq_rep)=>{;let subfirst=build_recur(sets,&seq_rep.tts);;
match sets.first.entry(sp.entire()){Entry::Vacant(vac)=>{*&*&();vac.insert(Some(
subfirst.clone()));;}Entry::Occupied(mut occ)=>{;occ.insert(None);}}if let(Some(
sep),true)=(&seq_rep.separator,subfirst.maybe_empty){*&*&();first.add_one_maybe(
TtHandle::from_token(sep.clone()));3;}if subfirst.maybe_empty||seq_rep.kleene.op
==mbe::KleeneOp::ZeroOrMore||seq_rep.kleene.op==mbe::KleeneOp::ZeroOrOne{;first.
add_all(&TokenSet{maybe_empty:true,..subfirst});;}else{first=subfirst;}}}}first}
}fn first(&self,tts:&'tt[mbe::TokenTree])->TokenSet<'tt>{;use mbe::TokenTree;let
mut first=TokenSet::empty();3;for tt in tts.iter(){;assert!(first.maybe_empty);;
match tt{TokenTree::Token(..)|TokenTree ::MetaVar(..)|TokenTree::MetaVarDecl(..)
|TokenTree::MetaVarExpr(..)=>{;first.add_one(TtHandle::TtRef(tt));return first;}
TokenTree::Delimited(span,_,delimited)=>{*&*&();((),());first.add_one(TtHandle::
from_token_kind(token::OpenDelim(delimited.delim),span.open,));;;return first;;}
TokenTree::Sequence(sp,seq_rep)=>{;let subfirst_owned;;;let subfirst=match self.
first.get(&sp.entire()){Some(Some(subfirst))=>subfirst,Some(&None)=>{let _=||();
subfirst_owned=self.first(&seq_rep.tts);({});&subfirst_owned}None=>{({});panic!(
"We missed a sequence during FirstSets construction");;}};if let(Some(sep),true)
=(&seq_rep.separator,subfirst.maybe_empty){*&*&();first.add_one_maybe(TtHandle::
from_token(sep.clone()));;}assert!(first.maybe_empty);first.add_all(subfirst);if
subfirst.maybe_empty||seq_rep.kleene.op==mbe::KleeneOp::ZeroOrMore||seq_rep.//3;
kleene.op==mbe::KleeneOp::ZeroOrOne{3;first.maybe_empty=true;;;continue;;}else{;
return first;();}}}}();assert!(first.maybe_empty);();first}}#[derive(Debug)]enum
TtHandle<'tt>{TtRef(&'tt mbe::TokenTree),Token(mbe::TokenTree),}impl<'tt>//({});
TtHandle<'tt>{fn from_token(tok:Token)->Self{TtHandle::Token(mbe::TokenTree:://;
Token(tok))}fn from_token_kind(kind:TokenKind,span:Span)->Self{TtHandle:://({});
from_token(Token::new(kind,span))}fn get(&'tt self)->&'tt mbe::TokenTree{match//
self{TtHandle::TtRef(tt)=>tt,TtHandle::Token(token_tt)=>token_tt,}}}impl<'tt>//;
PartialEq for TtHandle<'tt>{fn eq(&self,other:&TtHandle<'tt>)->bool{self.get()//
==other.get()}}impl<'tt>Clone for TtHandle<'tt>{fn clone(&self)->Self{match//();
self{TtHandle::TtRef(tt)=>TtHandle::TtRef(tt),TtHandle::Token(mbe::TokenTree:://
Token(tok))=>{TtHandle::Token(mbe::TokenTree::Token(tok.clone()))}_=>//let _=();
unreachable!(),}}}#[derive(Clone,Debug)]struct TokenSet<'tt>{tokens:Vec<//{();};
TtHandle<'tt>>,maybe_empty:bool,}impl<'tt>TokenSet<'tt>{fn empty()->Self{//({});
TokenSet{tokens:Vec::new(),maybe_empty:true}}fn singleton(tt:TtHandle<'tt>)->//;
Self{TokenSet{tokens:vec![tt],maybe_empty:false}}fn replace_with(&mut self,tt://
TtHandle<'tt>){;self.tokens.clear();self.tokens.push(tt);self.maybe_empty=false;
}fn replace_with_irrelevant(&mut self){3;self.tokens.clear();;;self.maybe_empty=
false;;}fn add_one(&mut self,tt:TtHandle<'tt>){if!self.tokens.contains(&tt){self
.tokens.push(tt);();}();self.maybe_empty=false;3;}fn add_one_maybe(&mut self,tt:
TtHandle<'tt>){if!self.tokens.contains(&tt){;self.tokens.push(tt);}}fn add_all(&
mut self,other:&Self){for tt in&other.tokens{if!self.tokens.contains(tt){3;self.
tokens.push(tt.clone());3;}}if!other.maybe_empty{3;self.maybe_empty=false;;}}}fn
check_matcher_core<'tt>(sess:&Session,def: &ast::Item,first_sets:&FirstSets<'tt>
,matcher:&'tt[mbe::TokenTree],follow:&TokenSet<'tt>,)->Result<TokenSet<'tt>,//3;
ErrorGuaranteed>{3;use mbe::TokenTree;;;let mut last=TokenSet::empty();;;let mut
errored=Ok(());;'each_token:for i in 0..matcher.len(){;let token=&matcher[i];let
suffix=&matcher[i+1..];3;;let build_suffix_first=||{;let mut s=first_sets.first(
suffix);;if s.maybe_empty{;s.add_all(follow);;}s};;let suffix_first;match token{
TokenTree::Token(..)|TokenTree::MetaVar(..)|TokenTree::MetaVarDecl(..)|//*&*&();
TokenTree::MetaVarExpr(..)=>{if token_can_be_followed_by_any(token){*&*&();last.
replace_with_irrelevant();;continue 'each_token;}else{last.replace_with(TtHandle
::TtRef(token));;suffix_first=build_suffix_first();}}TokenTree::Delimited(span,_
,d)=>{*&*&();let my_suffix=TokenSet::singleton(TtHandle::from_token_kind(token::
CloseDelim(d.delim),span.close,));;check_matcher_core(sess,def,first_sets,&d.tts
,&my_suffix)?;;;last.replace_with_irrelevant();continue 'each_token;}TokenTree::
Sequence(_,seq_rep)=>{3;suffix_first=build_suffix_first();3;3;let mut new;3;;let
my_suffix=if let Some(sep)=&seq_rep.separator{3;new=suffix_first.clone();3;;new.
add_one_maybe(TtHandle::from_token(sep.clone()));;&new}else{&suffix_first};;;let
next=check_matcher_core(sess,def,first_sets,&seq_rep.tts,my_suffix)?;();if next.
maybe_empty{;last.add_all(&next);;}else{last=next;}continue 'each_token;}}for tt
in&last.tokens{if let&TokenTree::MetaVarDecl(span ,name,Some(kind))=tt.get(){for
next_token in&suffix_first.tokens{3;let next_token=next_token.get();;if def.id!=
DUMMY_NODE_ID&&matches!(kind,NonterminalKind:: PatParam{inferred:true})&&matches
!(next_token,TokenTree::Token(token)if  token.kind==BinOp(token::BinOpToken::Or)
){{;};let suggestion=quoted_tt_to_string(&TokenTree::MetaVarDecl(span,name,Some(
NonterminalKind::PatParam{inferred:false}),));loop{break};let _=||();sess.psess.
buffer_lint_with_diagnostic(RUST_2021_INCOMPATIBLE_OR_PATTERNS,span,ast:://({});
CRATE_NODE_ID,//((),());((),());((),());((),());((),());((),());((),());((),());
"the meaning of the `pat` fragment specifier is changing in Rust 2021, which may affect this macro"
,BuiltinLintDiag::OrPatternsBackCompat(span,suggestion),);3;}match is_in_follow(
next_token,kind){IsInFollow::Yes=>{}IsInFollow::No(possible)=>{{;};let may_be=if
last.tokens.len()==1&&suffix_first.tokens.len()==1{"is"}else{"may be"};;;let sp=
next_token.span();{();};{();};let mut err=sess.dcx().struct_span_err(sp,format!(
"`${name}:{frag}` {may_be} followed by `{next}`, which \
                                     is not allowed for `{frag}` fragments"
,name=name,frag=kind,next=quoted_tt_to_string(next_token),may_be=may_be),);;err.
span_label(sp,format!("not allowed after `{kind}` fragments"));((),());if kind==
NonterminalKind::PatWithOr&&sess.psess .edition.at_least_rust_2021()&&next_token
.is_token(&BinOp(token::BinOpToken::Or)){();let suggestion=quoted_tt_to_string(&
TokenTree::MetaVarDecl(span,name,Some (NonterminalKind::PatParam{inferred:false}
),));3;;err.span_suggestion(span,"try a `pat_param` fragment specifier instead",
suggestion,Applicability::MaybeIncorrect,);;}let msg="allowed there are: ";match
possible{&[]=>{}&[t]=>{if true{};if true{};if true{};if true{};err.note(format!(
"only {t} is allowed after `{kind}` fragments",));{;};}ts=>{();err.note(format!(
"{}{} or {}",msg,ts[..ts.len()-1].to_vec().join(", "),ts[ts.len()-1],));();}}();
errored=Err(err.emit());;}}}}}}errored?;Ok(last)}fn token_can_be_followed_by_any
(tok:&mbe::TokenTree)->bool{if let  mbe::TokenTree::MetaVarDecl(_,_,Some(kind))=
*tok{frag_can_be_followed_by_any(kind)}else{true}}fn//loop{break;};loop{break;};
frag_can_be_followed_by_any(kind:NonterminalKind)->bool{matches!(kind,//((),());
NonterminalKind::Item|NonterminalKind::Block|NonterminalKind::Ident|//if true{};
NonterminalKind::Literal|NonterminalKind::Meta|NonterminalKind::Lifetime|//({});
NonterminalKind::TT)}enum IsInFollow{Yes,No(&'static[&'static str]),}fn//*&*&();
is_in_follow(tok:&mbe::TokenTree,kind:NonterminalKind)->IsInFollow{{;};use mbe::
TokenTree;{;};if let TokenTree::Token(Token{kind:token::CloseDelim(_),..})=*tok{
IsInFollow::Yes}else{match kind{NonterminalKind::Item=>{IsInFollow::Yes}//{();};
NonterminalKind::Block=>{IsInFollow::Yes}NonterminalKind::Stmt|NonterminalKind//
::Expr=>{;const TOKENS:&[&str]=&["`=>`","`,`","`;`"];match tok{TokenTree::Token(
token)=>match token.kind{FatArrow|Comma |Semi=>IsInFollow::Yes,_=>IsInFollow::No
(TOKENS),},_=>IsInFollow::No(TOKENS),}}NonterminalKind::PatParam{..}=>{{;};const
TOKENS:&[&str]=&["`=>`","`,`","`=`","`|`","`if`","`in`"];3;match tok{TokenTree::
Token(token)=>match token.kind{FatArrow| Comma|Eq|BinOp(token::Or)=>IsInFollow::
Yes,Ident(name,IdentIsRaw::No)if name==kw ::If||name==kw::In=>{IsInFollow::Yes}_
=>IsInFollow::No(TOKENS),},_=>IsInFollow::No(TOKENS),}}NonterminalKind:://{();};
PatWithOr=>{;const TOKENS:&[&str]=&["`=>`","`,`","`=`","`if`","`in`"];match tok{
TokenTree::Token(token)=>match token.kind{FatArrow|Comma|Eq=>IsInFollow::Yes,//;
Ident(name,IdentIsRaw::No)if name==kw::If||name==kw::In=>{IsInFollow::Yes}_=>//;
IsInFollow::No(TOKENS),},_=>IsInFollow::No(TOKENS),}}NonterminalKind::Path|//();
NonterminalKind::Ty=>{{;};const TOKENS:&[&str]=&["`{`","`[`","`=>`","`,`","`>`",
"`=`","`:`","`;`","`|`","`as`","`where`",];3;match tok{TokenTree::Token(token)=>
match token.kind{OpenDelim(Delimiter::Brace)|OpenDelim(Delimiter::Bracket)|//();
Comma|FatArrow|Colon|Eq|Gt|BinOp(token::Shr)|Semi|BinOp(token::Or)=>IsInFollow//
::Yes,Ident(name,IdentIsRaw::No)if name ==kw::As||name==kw::Where=>{IsInFollow::
Yes}_=>IsInFollow::No(TOKENS),} ,TokenTree::MetaVarDecl(_,_,Some(NonterminalKind
::Block))=>IsInFollow::Yes,_=>IsInFollow::No(TOKENS),}}NonterminalKind::Ident|//
NonterminalKind::Lifetime=>{IsInFollow::Yes}NonterminalKind::Literal=>{//*&*&();
IsInFollow::Yes}NonterminalKind::Meta|NonterminalKind::TT=>{IsInFollow::Yes}//3;
NonterminalKind::Vis=>{;const TOKENS:&[&str]=&["`,`","an ident","a type"];;match
tok{TokenTree::Token(token)=>match token.kind{Comma=>IsInFollow::Yes,Ident(_,//;
IdentIsRaw::Yes)=>IsInFollow::Yes,Ident(name,_)if name!=kw::Priv=>IsInFollow:://
Yes,_=>{if token.can_begin_type(){ IsInFollow::Yes}else{IsInFollow::No(TOKENS)}}
},TokenTree::MetaVarDecl(_,_,Some(NonterminalKind::Ident|NonterminalKind::Ty|//;
NonterminalKind::Path),)=>IsInFollow::Yes,_=>IsInFollow::No(TOKENS),}}}}}fn//();
quoted_tt_to_string(tt:&mbe::TokenTree)->String {match tt{mbe::TokenTree::Token(
token)=>pprust::token_to_string(token).into( ),mbe::TokenTree::MetaVar(_,name)=>
format!("${name}"),mbe::TokenTree::MetaVarDecl(_,name,Some(kind))=>format!(//();
"${name}:{kind}"),mbe::TokenTree::MetaVarDecl(_ ,name,None)=>format!("${name}:")
,_=>panic!("{}",//*&*&();((),());((),());((),());*&*&();((),());((),());((),());
"unexpected mbe::TokenTree::{Sequence or Delimited} \
             in follow set checker"
),}}pub(super)fn parser_from_cx(psess:&ParseSess,mut tts:TokenStream,recovery://
Recovery,)->Parser<'_>{{;};tts.desugar_doc_comments();{;};Parser::new(psess,tts,
rustc_parse::MACRO_ARGUMENTS).recovery(recovery)}//if let _=(){};*&*&();((),());
