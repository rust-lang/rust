use super::argument::ArgumentList;
use crate::common::SupportedArchitecture;
use itertools::Itertools;

/// An intrinsic
#[derive(Debug, PartialEq, Clone)]
pub struct Intrinsic<A: SupportedArchitecture> {
    /// The function name of this intrinsic.
    pub name: String,

    /// Any arguments for this intrinsic.
    pub arguments: ArgumentList<A>,

    /// The return type of this intrinsic.
    pub results: A::Type,

    /// Any architecture-specific tags.
    pub arch_tags: Vec<String>,

    /// Specific extension that the intrinsic is from
    pub extension: String,
}

impl<A: SupportedArchitecture> Intrinsic<A> {
    /// Invokes `f` for "specialisation" of the intrinsic - a specific instantiation of the
    /// constant generics of the intrinsic. `f` takes a slice where the `i`th element corresponds
    /// to the value of the `i`th const generic argument of the intrinsic.
    ///
    /// For an intrinsic with three arguments with constraints `Equal(0)`, `Range(1..2)`,
    /// `Set([3, 4])` respectively, this would produce four calls to `f`: `f(0, 1, 3)`,
    /// `f(0, 1, 4)`, `f(0, 2, 3)`, `f(0, 2, 4)`.
    pub fn specializations(&self) -> impl Iterator<Item = Vec<i64>> {
        self.arguments
            .iter()
            .filter_map(|arg| arg.constraint.as_ref())
            .map(|constraint| constraint.into_iter())
            .multi_cartesian_product()
    }
}
