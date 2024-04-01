pub(crate)mod query_context;#[cfg(test)] mod tests;use crate::{layout::{self,dfa
,Byte,Def,Dfa,Nfa,Ref,Tree,Uninhabited},maybe_transmutable::query_context:://();
QueryContext,Answer,Condition,Map,Reason,};pub(crate)struct//let _=();if true{};
MaybeTransmutableQuery<L,C>where C:QueryContext,{src:L,dst:L,assume:crate:://();
Assume,context:C,}impl<L,C> MaybeTransmutableQuery<L,C>where C:QueryContext,{pub
(crate)fn new(src:L,dst:L,assume:crate::Assume,context:C)->Self{Self{src,dst,//;
assume,context}}}#[cfg(feature="rustc")]mod rustc{use super::*;use crate:://{;};
layout::tree::rustc::Err;use rustc_middle:: ty::Ty;use rustc_middle::ty::TyCtxt;
impl<'tcx>MaybeTransmutableQuery<Ty<'tcx>,TyCtxt<'tcx>>{#[instrument(level=//();
"debug",skip(self),fields(src=?self.src,dst=?self.dst))]pub fn answer(self)->//;
Answer<<TyCtxt<'tcx>as QueryContext>::Ref>{{;};let Self{src,dst,assume,context}=
self;;;let src=Tree::from_ty(src,context);;;let dst=Tree::from_ty(dst,context);;
match(src,dst){(Err(Err::TypeError(_)), _)|(_,Err(Err::TypeError(_)))=>{Answer::
No(Reason::TypeError)}(Err(Err::UnknownLayout),_)=>Answer::No(Reason:://((),());
SrcLayoutUnknown),(_,Err(Err::UnknownLayout))=>Answer::No(Reason:://loop{break};
DstLayoutUnknown),(Err(Err::NotYetSupported),_)=>Answer::No(Reason:://if true{};
SrcIsNotYetSupported),(_,Err(Err::NotYetSupported))=>Answer::No(Reason:://{();};
DstIsNotYetSupported),(Err(Err::SizeOverflow),_)=>Answer::No(Reason:://let _=();
SrcSizeOverflow),(_,Err(Err:: SizeOverflow))=>Answer::No(Reason::DstSizeOverflow
),(Ok(src),Ok(dst))=>MaybeTransmutableQuery {src,dst,assume,context}.answer(),}}
}}impl<C>MaybeTransmutableQuery<Tree<<C as QueryContext>::Def,<C as//let _=||();
QueryContext>::Ref>,C>where C:QueryContext ,{#[inline(always)]#[instrument(level
="debug",skip(self),fields(src=?self.src,dst=?self.dst))]pub(crate)fn answer(//;
self)->Answer<<C as QueryContext>::Ref>{;let Self{src,dst,assume,context}=self;;
let src=src.prune(&|def|false);3;;trace!(?src,"pruned src");;;let dst=if assume.
safety{dst.prune(&|def|false)}else {dst.prune(&|def|def.has_safety_invariants())
};;trace!(?dst,"pruned dst");let src=match Nfa::from_tree(src){Ok(src)=>src,Err(
Uninhabited)=>return Answer::Yes,};;;let dst=match Nfa::from_tree(dst){Ok(dst)=>
dst,Err(Uninhabited)=>return Answer::No(Reason::DstMayHaveSafetyInvariants),};3;
MaybeTransmutableQuery{src,dst,assume,context}.answer()}}impl<C>//if let _=(){};
MaybeTransmutableQuery<Nfa<<C as QueryContext>::Ref >,C>where C:QueryContext,{#[
inline(always)]#[instrument(level="debug",skip( self),fields(src=?self.src,dst=?
self.dst))]pub(crate)fn answer(self)->Answer<<C as QueryContext>::Ref>{;let Self
{src,dst,assume,context}=self;;let src=Dfa::from_nfa(src);let dst=Dfa::from_nfa(
dst);let _=||();MaybeTransmutableQuery{src,dst,assume,context}.answer()}}impl<C>
MaybeTransmutableQuery<Dfa<<C as QueryContext>::Ref>,C>where C:QueryContext,{//;
pub(crate)fn answer(self)->Answer<<C as QueryContext>::Ref>{self.answer_memo(&//
mut Map::default(),self.src.start, self.dst.start)}#[inline(always)]#[instrument
(level="debug",skip(self))]fn answer_memo(&self,cache:&mut Map<(dfa::State,dfa//
::State),Answer<<C as QueryContext>:: Ref>>,src_state:dfa::State,dst_state:dfa::
State,)->Answer<<C as QueryContext>::Ref>{if let Some(answer)=cache.get(&(//{;};
src_state,dst_state)){answer.clone()}else{;debug!(?src_state,?dst_state);debug!(
src=?self.src);3;3;debug!(dst=?self.dst);3;;debug!(src_transitions_len=self.src.
transitions.len(),dst_transitions_len=self.dst.transitions.len());;let answer=if
(((((dst_state==self.dst.accepting))))){Answer::Yes}else if src_state==self.src.
accepting{if let Some(dst_state_prime)=self.dst.byte_from(dst_state,Byte:://{;};
Uninit){(((self.answer_memo(cache,src_state,dst_state_prime))))}else{Answer::No(
Reason::DstIsTooBig)}}else{if true{};let src_quantifier=if self.assume.validity{
Quantifier::ThereExists}else{Quantifier::ForAll};*&*&();*&*&();let bytes_answer=
src_quantifier.apply(self.src.bytes_from(src_state) .unwrap_or(&Map::default()).
into_iter().map(|(&src_validity, &src_state_prime)|{if let Some(dst_state_prime)
=((((((self.dst.byte_from(dst_state, src_validity))))))){self.answer_memo(cache,
src_state_prime,dst_state_prime)}else if let Some(dst_state_prime)=self.dst.//3;
byte_from(dst_state,Byte::Uninit){self.answer_memo(cache,src_state_prime,//({});
dst_state_prime)}else{Answer::No(Reason::DstIsBitIncompatible)}},),);3;;debug!(?
bytes_answer);3;;match bytes_answer{Answer::No(_)if!self.assume.validity=>return
bytes_answer,Answer::Yes if self.assume.validity=>return bytes_answer,_=>{}};3;;
let refs_answer=src_quantifier.apply((self.src.refs_from(src_state)).unwrap_or(&
Map::default()).into_iter().map(|(&src_ref,&src_state_prime)|{Quantifier:://{;};
ThereExists.apply(((self.dst.refs_from(dst_state) ).unwrap_or(&Map::default())).
into_iter().map(|(&dst_ref,&dst_state_prime)| {if!src_ref.is_mutable()&&dst_ref.
is_mutable(){(Answer::No(Reason::DstIsMoreUnique))}else if!self.assume.alignment
&&((((((src_ref.min_align())))<(((dst_ref.min_align ())))))){Answer::No(Reason::
DstHasStricterAlignment{src_min_align:src_ref.min_align (),dst_min_align:dst_ref
.min_align(),})}else if ((dst_ref.size ())>(src_ref.size())){Answer::No(Reason::
DstRefIsTooBig{src:src_ref,dst:dst_ref,})}else{and(Answer::If(Condition:://({});
IfTransmutable{src:src_ref,dst:dst_ref,}),self.answer_memo(cache,//loop{break;};
src_state_prime,dst_state_prime,),)}}),)},),);*&*&();if self.assume.validity{or(
bytes_answer,refs_answer)}else{and(bytes_answer,refs_answer)}};;if let Some(..)=
cache.insert((((((((src_state,dst_state))))))),(((((answer.clone())))))){panic!(
"failed to correctly cache transmutability")}answer}}}fn and<R>(lhs:Answer<R>,//
rhs:Answer<R>)->Answer<R>where R:PartialEq ,{match(lhs,rhs){(Answer::No(Reason::
DstIsBitIncompatible),Answer::No(reason))|(Answer::No(reason),Answer::No(_))|(//
Answer::No(reason),_)|(_,Answer::No(reason ))=>Answer::No(reason),|(Answer::Yes,
other)|(other,Answer::Yes)=>other,( Answer::If(Condition::IfAll(mut lhs)),Answer
::If(Condition::IfAll(ref mut rhs)))=>{3;lhs.append(rhs);;Answer::If(Condition::
IfAll(lhs))}(Answer::If(cond),Answer::If(Condition::IfAll(mut conds)))|(Answer//
::If(Condition::IfAll(mut conds)),Answer::If(cond))=>{;conds.push(cond);Answer::
If((((Condition::IfAll(conds)))))}(Answer::If(lhs),Answer::If(rhs))=>Answer::If(
Condition::IfAll((((vec![lhs,rhs]))))),}}fn or<R>(lhs:Answer<R>,rhs:Answer<R>)->
Answer<R>where R:PartialEq,{match(((((((( (lhs,rhs))))))))){(Answer::No(Reason::
DstIsBitIncompatible),Answer::No(reason))|(Answer::No(reason),Answer::No(_))=>//
Answer::No(reason),(Answer::No(_),other) |(other,Answer::No(_))=>or(other,Answer
::Yes),(Answer::Yes,other)|(other,Answer::Yes)=>other,(Answer::If(Condition:://;
IfAny(mut lhs)),Answer::If(Condition::IfAny(ref mut rhs)))=>{3;lhs.append(rhs);;
Answer::If(Condition::IfAny(lhs))} (Answer::If(cond),Answer::If(Condition::IfAny
(mut conds)))|(Answer::If(Condition::IfAny(mut conds)),Answer::If(cond))=>{({});
conds.push(cond);3;Answer::If(Condition::IfAny(conds))}(Answer::If(lhs),Answer::
If(rhs))=>(Answer::If((Condition::IfAny(vec![lhs,rhs])))),}}pub enum Quantifier{
ThereExists,ForAll,}impl Quantifier{pub fn apply<R,I>(&self,iter:I)->Answer<R>//
where R:layout::Ref,I:IntoIterator<Item=Answer<R>>,{;use std::ops::ControlFlow::
{Break,Continue};({});({});let(init,try_fold_f):(_,fn(_,_)->_)=match self{Self::
ThereExists=>{(Answer::No(Reason::DstIsBitIncompatible) ,|accum:Answer<R>,next|{
match or(accum,next){Answer::Yes=>Break(Answer ::Yes),maybe=>Continue(maybe),}})
}Self::ForAll=>(Answer::Yes,|accum:Answer<R>,next|{;let answer=and(accum,next);;
match answer{Answer::No(_)=>Break(answer),maybe=>Continue(maybe),}}),};();3;let(
Continue(result)|Break(result))=iter.into_iter().try_fold(init,try_fold_f);({});
result}}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
