use super::Goal;
use crate::Interner;

/// An obligation that can be processed by the shared fulfillment engine.
///
/// The shared engine only accesses the parts needed for fulfillment through this trait.
pub trait FulfillmentObligation<I: Interner>: Clone {
    fn as_goal(&self) -> Goal<I, I::Predicate>;

    fn span(&self) -> I::Span;

    fn recursion_depth(&self) -> usize;

    fn set_recursion_depth(&mut self, depth: usize);

    /// Stores the eagerly resolved predicate returned by the solver so that
    /// later fulfillment iterations do not repeat that work.
    fn set_predicate(&mut self, predicate: I::Predicate);
}
