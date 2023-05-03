use rustc_span::Span;

use crate::ty::Predicate;

/// Represents the bounds declared on a particular set of type
/// parameters. Should eventually be generalized into a flag list of
/// where-clauses. You can obtain an `InstantiatedPredicates` list from a
/// `GenericPredicates` by using the `instantiate` method. Note that this method
/// reflects an important semantic invariant of `InstantiatedPredicates`: while
/// the `GenericPredicates` are expressed in terms of the bound type
/// parameters of the impl/trait/whatever, an `InstantiatedPredicates` instance
/// represented a set of bounds for some particular instantiation,
/// meaning that the generic parameters have been substituted with
/// their values.
///
/// Example:
/// ```ignore (illustrative)
/// struct Foo<T, U: Bar<T>> { ... }
/// ```
/// Here, the `GenericPredicates` for `Foo` would contain a list of bounds like
/// `[[], [U:Bar<T>]]`. Now if there were some particular reference
/// like `Foo<isize,usize>`, then the `InstantiatedPredicates` would be `[[],
/// [usize:Bar<isize>]]`.
#[derive(Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct InstantiatedPredicates<'tcx> {
    pub predicates: Vec<Predicate<'tcx>>,
    pub spans: Vec<Span>,
}

impl<'tcx> InstantiatedPredicates<'tcx> {
    pub fn empty() -> InstantiatedPredicates<'tcx> {
        InstantiatedPredicates { predicates: vec![], spans: vec![] }
    }

    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }

    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        (&self).into_iter()
    }
}

impl<'tcx> IntoIterator for InstantiatedPredicates<'tcx> {
    type Item = (Predicate<'tcx>, Span);

    type IntoIter = std::iter::Zip<std::vec::IntoIter<Predicate<'tcx>>, std::vec::IntoIter<Span>>;

    fn into_iter(self) -> Self::IntoIter {
        debug_assert_eq!(self.predicates.len(), self.spans.len());
        std::iter::zip(self.predicates, self.spans)
    }
}

impl<'a, 'tcx> IntoIterator for &'a InstantiatedPredicates<'tcx> {
    type Item = (Predicate<'tcx>, Span);

    type IntoIter = std::iter::Zip<
        std::iter::Copied<std::slice::Iter<'a, Predicate<'tcx>>>,
        std::iter::Copied<std::slice::Iter<'a, Span>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        debug_assert_eq!(self.predicates.len(), self.spans.len());
        std::iter::zip(self.predicates.iter().copied(), self.spans.iter().copied())
    }
}
