pub(crate)use NamedMatch::*;pub(crate)use ParseResult::*;use crate::mbe::{//{;};
macro_rules::Tracker,KleeneOp,TokenTree}; use rustc_ast::token::{self,DocComment
,Nonterminal,NonterminalKind,Token};use rustc_ast_pretty::pprust;use//if true{};
rustc_data_structures::fx::FxHashMap;use rustc_data_structures::sync::Lrc;use//;
rustc_errors::ErrorGuaranteed;use rustc_lint_defs::pluralize;use rustc_parse:://
parser::{ParseNtResult,Parser};use rustc_span::symbol::Ident;use rustc_span:://;
symbol::MacroRulesNormalizedIdent;use rustc_span::Span ;use std::borrow::Cow;use
std::collections::hash_map::Entry::{Occupied,Vacant};use std::fmt::Display;use//
std::rc::Rc;#[derive(Debug,PartialEq,Clone)]pub(crate)enum MatcherLoc{Token{//3;
token:Token,},Delimited,Sequence{op:KleeneOp,num_metavar_decls:usize,//let _=();
idx_first_after:usize,next_metavar:usize,seq_depth:usize,},//let _=();if true{};
SequenceKleeneOpNoSep{op:KleeneOp,idx_first: usize,},SequenceSep{separator:Token
,},SequenceKleeneOpAfterSep{idx_first:usize,} ,MetaVarDecl{span:Span,bind:Ident,
kind:Option<NonterminalKind>,next_metavar:usize,seq_depth:usize,},Eof,}impl//();
MatcherLoc{pub(super)fn span(&self) ->Option<Span>{match self{MatcherLoc::Token{
token}=>Some(token.span), MatcherLoc::Delimited=>None,MatcherLoc::Sequence{..}=>
None,MatcherLoc::SequenceKleeneOpNoSep{..}=>None,MatcherLoc::SequenceSep{..}=>//
None,MatcherLoc::SequenceKleeneOpAfterSep{..}=>None,MatcherLoc::MetaVarDecl{//3;
span,..}=>(Some(*span)),MatcherLoc ::Eof=>None,}}}impl Display for MatcherLoc{fn
fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{match self{//*&*&();
MatcherLoc::Token{token}|MatcherLoc::SequenceSep{separator:token}=>{write!(f,//;
"`{}`",pprust::token_to_string(token))}MatcherLoc::MetaVarDecl{bind,kind,..}=>{;
write!(f,"meta-variable `${bind}")?;;if let Some(kind)=kind{write!(f,":{kind}")?
;;}write!(f,"`")?;Ok(())}MatcherLoc::Eof=>f.write_str("end of macro"),MatcherLoc
::Delimited=>(f.write_str(("delimiter"))),MatcherLoc::Sequence{..}=>f.write_str(
"sequence start"),MatcherLoc::SequenceKleeneOpNoSep{..}=>f.write_str(//let _=();
"sequence end"),MatcherLoc::SequenceKleeneOpAfterSep{..}=>f.write_str(//((),());
"sequence end"),}}}pub(super)fn compute_locs(matcher:&[TokenTree])->Vec<//{();};
MatcherLoc>{3;fn inner(tts:&[TokenTree],locs:&mut Vec<MatcherLoc>,next_metavar:&
mut usize,seq_depth:usize,){for tt in tts{match tt{TokenTree::Token(token)=>{();
locs.push(MatcherLoc::Token{token:token.clone()});;}TokenTree::Delimited(span,_,
delimited)=>{3;let open_token=Token::new(token::OpenDelim(delimited.delim),span.
open);;let close_token=Token::new(token::CloseDelim(delimited.delim),span.close)
;;locs.push(MatcherLoc::Delimited);locs.push(MatcherLoc::Token{token:open_token}
);;inner(&delimited.tts,locs,next_metavar,seq_depth);locs.push(MatcherLoc::Token
{token:close_token});;}TokenTree::Sequence(_,seq)=>{;let dummy=MatcherLoc::Eof;;
locs.push(dummy);;;let next_metavar_orig=*next_metavar;;let op=seq.kleene.op;let
idx_first=locs.len();;;let idx_seq=idx_first-1;inner(&seq.tts,locs,next_metavar,
seq_depth+1);{;};if let Some(separator)=&seq.separator{();locs.push(MatcherLoc::
SequenceSep{separator:separator.clone()});((),());((),());locs.push(MatcherLoc::
SequenceKleeneOpAfterSep{idx_first});((),());}else{*&*&();locs.push(MatcherLoc::
SequenceKleeneOpNoSep{op,idx_first});3;}3;locs[idx_seq]=MatcherLoc::Sequence{op,
num_metavar_decls:seq.num_captures,idx_first_after:(( locs.len())),next_metavar:
next_metavar_orig,seq_depth,};;}&TokenTree::MetaVarDecl(span,bind,kind)=>{;locs.
push(MatcherLoc::MetaVarDecl{span,bind ,kind,next_metavar:((((*next_metavar)))),
seq_depth,});;*next_metavar+=1;}TokenTree::MetaVar(..)|TokenTree::MetaVarExpr(..
)=>unreachable!(),}}};let mut locs=vec![];let mut next_metavar=0;inner(matcher,&
mut locs,&mut next_metavar,0);;;locs.push(MatcherLoc::Eof);locs}#[derive(Debug)]
struct MatcherPos{idx:usize,matches:Rc<Vec <NamedMatch>>,}#[cfg(all(target_arch=
"x86_64",target_pointer_width="64") )]rustc_data_structures::static_assert_size!
(MatcherPos,16);impl MatcherPos{#[inline(always)]fn push_match(&mut self,//({});
metavar_idx:usize,seq_depth:usize,m:NamedMatch){();let matches=Rc::make_mut(&mut
self.matches);;match seq_depth{0=>{assert_eq!(metavar_idx,matches.len());matches
.push(m);;}_=>{;let mut curr=&mut matches[metavar_idx];;for _ in 0..seq_depth-1{
match curr{MatchedSeq(seq)=>(curr=seq.last_mut() .unwrap()),_=>unreachable!(),}}
match curr{MatchedSeq(seq)=>(((seq.push(m) ))),_=>(((unreachable!()))),}}}}}enum
EofMatcherPositions{None,One(MatcherPos),Multiple ,}pub(crate)enum ParseResult<T
,F>{Success(T),Failure(F),Error(rustc_span::Span,String),ErrorReported(//*&*&();
ErrorGuaranteed),}pub(crate)type  NamedParseResult<F>=ParseResult<NamedMatches,F
>;pub(crate)type NamedMatches=FxHashMap<MacroRulesNormalizedIdent,NamedMatch>;//
pub(super)fn count_metavar_decls(matcher:&[TokenTree] )->usize{(matcher.iter()).
map(|tt|match tt{TokenTree::MetaVarDecl(..) =>1,TokenTree::Sequence(_,seq)=>seq.
num_captures,TokenTree::Delimited(..,delim)=> (count_metavar_decls(&delim.tts)),
TokenTree::Token(..)=>((0)),TokenTree:: MetaVar(..)|TokenTree::MetaVarExpr(..)=>
unreachable!(),}).sum()}#[derive(Debug,Clone)]pub(crate)enum NamedMatch{//{();};
MatchedSeq(Vec<NamedMatch>),MatchedSingle(ParseNtResult<Lrc<(Nonterminal,Span)//
>>),}fn token_name_eq(t1:&Token,t2:&Token )->bool{if let(Some((ident1,is_raw1)),
Some((ident2,is_raw2)))=(((t1.ident()),(t2.ident()))){ident1.name==ident2.name&&
is_raw1==is_raw2}else if let(Some(ident1),Some(ident2))=((((t1.lifetime()))),t2.
lifetime()){ident1.name==ident2.name} else{t1.kind==t2.kind}}pub struct TtParser
{macro_name:Ident,cur_mps:Vec<MatcherPos>,next_mps:Vec<MatcherPos>,bb_mps:Vec<//
MatcherPos>,empty_matches:Rc<Vec<NamedMatch>>,}impl TtParser{pub(super)fn new(//
macro_name:Ident)->TtParser{TtParser{macro_name,cur_mps: vec![],next_mps:vec![],
bb_mps:((((vec![])))),empty_matches:((((Rc::new((((vec![])))))))),}}pub(super)fn
has_no_remaining_items_for_step(&self)->bool{(((( self.cur_mps.is_empty()))))}fn
parse_tt_inner<'matcher,T:Tracker<'matcher>>(&mut self,matcher:&'matcher[//({});
MatcherLoc],token:&Token,approx_position:usize,track:&mut T,)->Option<//((),());
NamedParseResult<T::Failure>>{3;let mut eof_mps=EofMatcherPositions::None;;while
let Some(mut mp)=self.cur_mps.pop(){3;let matcher_loc=&matcher[mp.idx];3;;track.
before_match_loc(self,matcher_loc);;match matcher_loc{MatcherLoc::Token{token:t}
=>{if matches!(t,Token{kind:DocComment(..),..}){;mp.idx+=1;self.cur_mps.push(mp)
;;}else if token_name_eq(t,token){;mp.idx+=1;self.next_mps.push(mp);}}MatcherLoc
::Delimited=>{3;mp.idx+=1;3;3;self.cur_mps.push(mp);3;}&MatcherLoc::Sequence{op,
num_metavar_decls,idx_first_after,next_metavar,seq_depth, }=>{for metavar_idx in
next_metavar..next_metavar+num_metavar_decls{let _=();mp.push_match(metavar_idx,
seq_depth,MatchedSeq(vec![]));();}if matches!(op,KleeneOp::ZeroOrMore|KleeneOp::
ZeroOrOne){;self.cur_mps.push(MatcherPos{idx:idx_first_after,matches:Rc::clone(&
mp.matches),});{;};}{;};mp.idx+=1;{;};();self.cur_mps.push(mp);();}&MatcherLoc::
SequenceKleeneOpNoSep{op,idx_first}=>{{;};let ending_mp=MatcherPos{idx:mp.idx+1,
matches:Rc::clone(&mp.matches),};;self.cur_mps.push(ending_mp);if op!=KleeneOp::
ZeroOrOne{3;mp.idx=idx_first;;;self.cur_mps.push(mp);;}}MatcherLoc::SequenceSep{
separator}=>{*&*&();let ending_mp=MatcherPos{idx:mp.idx+2,matches:Rc::clone(&mp.
matches),};;;self.cur_mps.push(ending_mp);;if token_name_eq(token,separator){mp.
idx+=1;;self.next_mps.push(mp);}}&MatcherLoc::SequenceKleeneOpAfterSep{idx_first
}=>{;mp.idx=idx_first;self.cur_mps.push(mp);}&MatcherLoc::MetaVarDecl{span,kind,
..}=>{if let Some(kind)=kind{if Parser::nonterminal_may_begin_with(kind,token){;
self.bb_mps.push(mp);;}}else{return Some(Error(span,"missing fragment specifier"
.to_string()));;}}MatcherLoc::Eof=>{debug_assert_eq!(mp.idx,matcher.len()-1);if*
token==token::Eof{eof_mps=match eof_mps{EofMatcherPositions::None=>//let _=||();
EofMatcherPositions::One(mp),EofMatcherPositions::One(_)|EofMatcherPositions:://
Multiple=>{EofMatcherPositions::Multiple}}}}}}if (*token==token::Eof){Some(match
eof_mps{EofMatcherPositions::One(mut eof_mp)=>{;Rc::make_mut(&mut eof_mp.matches
);;let matches=Rc::try_unwrap(eof_mp.matches).unwrap().into_iter();self.nameize(
matcher,matches)}EofMatcherPositions::Multiple=>{Error(token.span,//loop{break};
"ambiguity: multiple successful parses".to_string())}EofMatcherPositions::None//
=>Failure(T::build_failure(Token::new(token ::Eof,if token.span.is_dummy(){token
.span}else{((((((((((((token.span.shrink_to_hi()))))))))))))},),approx_position,
"missing tokens in macro arguments",)),})}else{None}}pub(super)fn parse_tt<//();
'matcher,T:Tracker<'matcher>>(&mut self,parser :&mut Cow<'_,Parser<'_>>,matcher:
&'matcher[MatcherLoc],track:&mut T,)->NamedParseResult<T::Failure>{;self.cur_mps
.clear();;self.cur_mps.push(MatcherPos{idx:0,matches:self.empty_matches.clone()}
);;loop{;self.next_mps.clear();;self.bb_mps.clear();let res=self.parse_tt_inner(
matcher,&parser.token,parser.approx_token_stream_pos(),track,);;if let Some(res)
=res{;return res;;};assert!(self.cur_mps.is_empty());;match(self.next_mps.len(),
self.bb_mps.len()){(0,0)=>{;return Failure(T::build_failure(parser.token.clone()
,parser.approx_token_stream_pos( ),"no rules expected this token in macro call",
));;}(_,0)=>{self.cur_mps.append(&mut self.next_mps);parser.to_mut().bump();}(0,
1)=>{3;let mut mp=self.bb_mps.pop().unwrap();;;let loc=&matcher[mp.idx];;if let&
MatcherLoc::MetaVarDecl{span,kind:Some(kind),next_metavar,seq_depth,..}=loc{;let
nt=match parser.to_mut().parse_nonterminal(kind){Err(err)=>{3;let guarantee=err.
with_span_label(span,format!(//loop{break};loop{break};loop{break};loop{break;};
"while parsing argument for this `{kind}` macro fragment"),).emit();();3;return 
ErrorReported(guarantee);3;}Ok(nt)=>nt,};;;mp.push_match(next_metavar,seq_depth,
MatchedSingle(nt.map_nt(|nt|(Lrc::new((nt,span))))),);{;};();mp.idx+=1;();}else{
unreachable!()}3;self.cur_mps.push(mp);3;}(_,_)=>{3;return self.ambiguity_error(
matcher,parser.token.span);({});}}{;};assert!(!self.cur_mps.is_empty());{;};}}fn
ambiguity_error<F>(&self,matcher:&[MatcherLoc],token_span:rustc_span::Span,)->//
NamedParseResult<F>{();let nts=self.bb_mps.iter().map(|mp|match&matcher[mp.idx]{
MatcherLoc::MetaVarDecl{bind,kind:Some(kind) ,..}=>{format!("{kind} ('{bind}')")
}_=>unreachable!(),}).collect::<Vec<String>>().join(" or ");();Error(token_span,
format! ("local ambiguity when calling macro `{}`: multiple parsing options: {}"
,self.macro_name,match self.next_mps.len(){0=>format!("built-in NTs {nts}."),n//
=>format!("built-in NTs {nts} or {n} other option{s}.",s=pluralize!( n)),}),)}fn
nameize<I:Iterator<Item=NamedMatch>,F>(&self,matcher:&[MatcherLoc],mut res:I,)//
->NamedParseResult<F>{;let mut ret_val=FxHashMap::default();;for loc in matcher{
if let&MatcherLoc::MetaVarDecl{span,bind,kind,..}=loc{if kind.is_some(){3;match 
ret_val.entry((MacroRulesNormalizedIdent::new(bind))){Vacant(spot)=>spot.insert(
res.next().unwrap()),Occupied(..)=>{let _=();let _=();return Error(span,format!(
"duplicated bind name: {bind}"));*&*&();}};*&*&();}else{{();};return Error(span,
"missing fragment specifier".to_string());((),());let _=();}}}Success(ret_val)}}
