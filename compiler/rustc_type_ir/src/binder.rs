use crate::Interner;

pub trait BoundVars<I: Interner> {
    fn bound_vars(&self) -> I::BoundVars;

    fn has_no_bound_vars(&self) -> bool;
}
