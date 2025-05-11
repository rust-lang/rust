//! The move-analysis portion of borrowck needs to work in an abstract
//! domain of lifted `Place`s. Most of the `Place` variants fall into a
//! one-to-one mapping between the concrete and abstract (e.g., a
//! field-deref on a local variable, `x.field`, has the same meaning
//! in both domains). Indexed projections are the exception: `a[x]`
//! needs to be treated as mapping to the same move path as `a[y]` as
//! well as `a[13]`, etc. So we map these `x`/`y` values to `()`.
//!
//! (In theory, the analysis could be extended to work with sets of
//! paths, so that `a[0]` and `a[13]` could be kept distinct, while
//! `a[x]` would still overlap them both. But that is not this
//! representation does today.)

use rustc_middle::mir::{PlaceElem, ProjectionElem, ProjectionKind};

pub(crate) trait Lift {
    fn lift(&self) -> ProjectionKind;
}

impl<'tcx> Lift for PlaceElem<'tcx> {
    fn lift(&self) -> ProjectionKind {
        match *self {
            ProjectionElem::Deref => ProjectionElem::Deref,
            ProjectionElem::Field(f, _ty) => ProjectionElem::Field(f, ()),
            ProjectionElem::OpaqueCast(_ty) => ProjectionElem::OpaqueCast(()),
            ProjectionElem::Index(_i) => ProjectionElem::Index(()),
            ProjectionElem::Subslice { from, to, from_end } => {
                ProjectionElem::Subslice { from, to, from_end }
            }
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                ProjectionElem::ConstantIndex { offset, min_length, from_end }
            }
            ProjectionElem::Downcast(a, u) => ProjectionElem::Downcast(a, u),
            ProjectionElem::Subtype(_ty) => ProjectionElem::Subtype(()),
            ProjectionElem::UnwrapUnsafeBinder(_ty) => ProjectionElem::UnwrapUnsafeBinder(()),
        }
    }
}
