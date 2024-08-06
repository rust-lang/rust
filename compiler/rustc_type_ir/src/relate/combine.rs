use super::TypeRelation;
use crate::solve::Goal;
use crate::{InferCtxtLike, Interner, Upcast};

pub trait PredicateEmittingRelation<Infcx, I = <Infcx as InferCtxtLike>::Interner>:
    TypeRelation<I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    fn span(&self) -> I::Span;

    fn param_env(&self) -> I::ParamEnv;

    /// Register obligations that must hold in order for this relation to hold
    fn register_goals(&mut self, obligations: impl IntoIterator<Item = Goal<I, I::Predicate>>);

    /// Register predicates that must hold in order for this relation to hold.
    /// This uses the default `param_env` of the obligation.
    fn register_predicates(
        &mut self,
        obligations: impl IntoIterator<Item: Upcast<I, I::Predicate>>,
    );

    /// Register `AliasRelate` obligation(s) that both types must be related to each other.
    fn register_alias_relate_predicate(&mut self, a: I::Ty, b: I::Ty);
}
