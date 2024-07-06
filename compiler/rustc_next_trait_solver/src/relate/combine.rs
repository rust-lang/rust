pub use rustc_type_ir::relate::*;
use rustc_type_ir::solve::Goal;
use rustc_type_ir::{InferCtxtLike, Interner, Upcast};

use super::StructurallyRelateAliases;

pub trait PredicateEmittingRelation<Infcx, I = <Infcx as InferCtxtLike>::Interner>:
    TypeRelation<I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    fn span(&self) -> I::Span;

    fn param_env(&self) -> I::ParamEnv;

    /// Whether aliases should be related structurally. This is pretty much
    /// always `No` unless you're equating in some specific locations of the
    /// new solver. See the comments in these use-cases for more details.
    fn structurally_relate_aliases(&self) -> StructurallyRelateAliases;

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
