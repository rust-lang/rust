// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The move-analysis portion of borrowck needs to work in an abstract
//! domain of lifted Lvalues.  Most of the Lvalue variants fall into a
//! one-to-one mapping between the concrete and abstract (e.g. a
//! field-deref on a local-variable, `x.field`, has the same meaning
//! in both domains). Indexed-Projections are the exception: `a[x]`
//! needs to be treated as mapping to the same move path as `a[y]` as
//! well as `a[13]`, et cetera.
//!
//! (In theory the analysis could be extended to work with sets of
//! paths, so that `a[0]` and `a[13]` could be kept distinct, while
//! `a[x]` would still overlap them both. But that is not this
//! representation does today.)

use rustc::mir::LvalueElem;
use rustc::mir::{Operand, ProjectionElem};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct AbstractOperand;
pub type AbstractElem<'tcx> =
    ProjectionElem<'tcx, AbstractOperand>;

pub trait Lift {
    type Abstract;
    fn lift(&self) -> Self::Abstract;
}
impl<'tcx> Lift for Operand<'tcx> {
    type Abstract = AbstractOperand;
    fn lift(&self) -> Self::Abstract { AbstractOperand }
}
impl<'tcx> Lift for LvalueElem<'tcx> {
    type Abstract = AbstractElem<'tcx>;
    fn lift(&self) -> Self::Abstract {
        match *self {
            ProjectionElem::Deref =>
                ProjectionElem::Deref,
            ProjectionElem::Field(ref f, ty) =>
                ProjectionElem::Field(f.clone(), ty.clone()),
            ProjectionElem::Index(ref i) =>
                ProjectionElem::Index(i.lift()),
            ProjectionElem::Subslice {from, to} =>
                ProjectionElem::Subslice { from: from, to: to },
            ProjectionElem::ConstantIndex {offset,min_length,from_end} =>
                ProjectionElem::ConstantIndex {
                    offset: offset,
                    min_length: min_length,
                    from_end: from_end
                },
            ProjectionElem::Downcast(a, u) =>
                ProjectionElem::Downcast(a.clone(), u.clone()),
        }
    }
}
