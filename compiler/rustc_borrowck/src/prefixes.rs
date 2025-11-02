//! From the NLL RFC:
//! "Shallow prefixes are found by stripping away fields, but stop at
//! any dereference. So: writing a path like `a` is illegal if `a.b`
//! is borrowed. But: writing `a` is legal if `*a` is borrowed,
//! whether or not `a` is a shared or mutable reference. [...] "

use rustc_middle::mir::{PlaceRef, ProjectionElem};

use super::MirBorrowckCtxt;

pub(crate) trait IsPrefixOf<'tcx> {
    fn is_prefix_of(&self, other: PlaceRef<'tcx>) -> bool;
}

impl<'tcx> IsPrefixOf<'tcx> for PlaceRef<'tcx> {
    fn is_prefix_of(&self, other: PlaceRef<'tcx>) -> bool {
        self.local == other.local
            && self.projection.len() <= other.projection.len()
            && self.projection == &other.projection[..self.projection.len()]
    }
}

pub(super) struct Prefixes<'tcx> {
    kind: PrefixSet,
    next: Option<PlaceRef<'tcx>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum PrefixSet {
    /// Doesn't stop until it returns the base case (a Local or
    /// Static prefix).
    All,
    /// Stops at any dereference.
    Shallow,
}

impl<'tcx> MirBorrowckCtxt<'_, '_, 'tcx> {
    /// Returns an iterator over the prefixes of `place`
    /// (inclusive) from longest to smallest, potentially
    /// terminating the iteration early based on `kind`.
    pub(super) fn prefixes(&self, place_ref: PlaceRef<'tcx>, kind: PrefixSet) -> Prefixes<'tcx> {
        Prefixes { next: Some(place_ref), kind }
    }
}

impl<'tcx> Iterator for Prefixes<'tcx> {
    type Item = PlaceRef<'tcx>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut cursor = self.next?;

        // Post-processing `place`: Enqueue any remaining
        // work. Also, `place` may not be a prefix itself, but
        // may hold one further down (e.g., we never return
        // downcasts here, but may return a base of a downcast).

        loop {
            match cursor.last_projection() {
                None => {
                    self.next = None;
                    return Some(cursor);
                }
                Some((cursor_base, elem)) => {
                    match elem {
                        ProjectionElem::Field(_ /*field*/, _ /*ty*/) => {
                            // FIXME: add union handling
                            self.next = Some(cursor_base);
                            return Some(cursor);
                        }
                        ProjectionElem::UnwrapUnsafeBinder(_) => {
                            self.next = Some(cursor_base);
                            return Some(cursor);
                        }
                        ProjectionElem::Downcast(..)
                        | ProjectionElem::Subslice { .. }
                        | ProjectionElem::OpaqueCast { .. }
                        | ProjectionElem::ConstantIndex { .. }
                        | ProjectionElem::Index(_) => {
                            cursor = cursor_base;
                        }
                        ProjectionElem::Deref => {
                            match self.kind {
                                PrefixSet::Shallow => {
                                    // Shallow prefixes are found by stripping away
                                    // fields, but stop at *any* dereference.
                                    // So we can just stop the traversal now.
                                    self.next = None;
                                    return Some(cursor);
                                }
                                PrefixSet::All => {
                                    // All prefixes: just blindly enqueue the base
                                    // of the projection.
                                    self.next = Some(cursor_base);
                                    return Some(cursor);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
