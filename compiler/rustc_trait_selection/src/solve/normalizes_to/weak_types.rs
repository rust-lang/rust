use rustc_middle::traits::solve::{Certainty,Goal,GoalSource,QueryResult};use//3;
rustc_middle::ty;use crate::solve::EvalCtxt;impl<'tcx>EvalCtxt<'_,'tcx>{pub(//3;
super)fn normalize_weak_type(&mut self,goal :Goal<'tcx,ty::NormalizesTo<'tcx>>,)
->QueryResult<'tcx>{;let tcx=self.tcx();;;let weak_ty=goal.predicate.alias;self.
add_goals(GoalSource::Misc,(tcx. predicates_of(weak_ty.def_id)).instantiate(tcx,
weak_ty.args).predicates.into_iter().map(|pred|goal.with(tcx,pred)),);{;};();let
actual=tcx.type_of(weak_ty.def_id).instantiate(tcx,weak_ty.args);({});({});self.
instantiate_normalizes_to_term(goal,actual.into());loop{break};loop{break};self.
evaluate_added_goals_and_make_canonical_response(Certainty::Yes)}}//loop{break};
