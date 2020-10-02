//! The move-analysis portion of borrowck needs to work in an abstract
//! domain of lifted `Place`s. Most of the `Place` variants fall into a
//! one-to-one mapping between the concrete and abstract (e.g., a
//! field-deref on a local variable, `x.field`, has the same meaning
//! in both domains). Indexed projections are the exception: `a[x]`
//! needs to be treated as mapping to the same move path as `a[y]` as
//! well as `a[13]`, etc.
//!
//! (In theory, the analysis could be extended to work with sets of
//! paths, so that `a[0]` and `a[13]` could be kept distinct, while
//! `a[x]` would still overlap them both. But that is not this
//! representation does today.)

use rustc_middle::mir::{Local, Operand, PlaceElem, ProjectionElem};
use rustc_middle::ty::Ty;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct AbstractOperand;
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct AbstractType;
pub type AbstractElem = ProjectionElem<AbstractOperand, AbstractType>;

pub trait Lift {
    type Abstract;
    fn lift(&self) -> Self::Abstract;
}
impl<'tcx> Lift for Operand<'tcx> {
    type Abstract = AbstractOperand;
    fn lift(&self) -> Self::Abstract {
        AbstractOperand
    }
}
impl Lift for Local {
    type Abstract = AbstractOperand;
    fn lift(&self) -> Self::Abstract {
        AbstractOperand
    }
}
impl<'tcx> Lift for Ty<'tcx> {
    type Abstract = AbstractType;
    fn lift(&self) -> Self::Abstract {
        AbstractType
    }
}
impl<'tcx> Lift for PlaceElem<'tcx> {
    type Abstract = AbstractElem;
    fn lift(&self) -> Self::Abstract {
        match *self {
            ProjectionElem::Deref => ProjectionElem::Deref,
            ProjectionElem::Field(f, ty) => ProjectionElem::Field(f, ty.lift()),
            ProjectionElem::Index(ref i) => ProjectionElem::Index(i.lift()),
            ProjectionElem::Subslice { from, to, from_end } => {
                ProjectionElem::Subslice { from, to, from_end }
            }
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                ProjectionElem::ConstantIndex { offset, min_length, from_end }
            }
            ProjectionElem::Downcast(a, u) => ProjectionElem::Downcast(a, u),
        }
    }
}
