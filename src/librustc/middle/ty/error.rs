// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def_id::DefId;
use middle::subst;
use middle::infer::type_variable;
use middle::ty::{self, BoundRegion, Region, Ty};

use std::fmt;
use syntax::abi;
use syntax::ast::{self, Name};
use syntax::codemap::Span;

use rustc_front::hir;

#[derive(Clone, Copy, Debug)]
pub struct ExpectedFound<T> {
    pub expected: T,
    pub found: T
}

// Data structures used in type unification
#[derive(Clone, Debug)]
pub enum TypeError<'tcx> {
    Mismatch,
    UnsafetyMismatch(ExpectedFound<hir::Unsafety>),
    AbiMismatch(ExpectedFound<abi::Abi>),
    Mutability,
    BoxMutability,
    PtrMutability,
    RefMutability,
    VecMutability,
    TupleSize(ExpectedFound<usize>),
    FixedArraySize(ExpectedFound<usize>),
    TyParamSize(ExpectedFound<usize>),
    ArgCount,
    RegionsDoesNotOutlive(Region, Region),
    RegionsNotSame(Region, Region),
    RegionsNoOverlap(Region, Region),
    RegionsInsufficientlyPolymorphic(BoundRegion, Region),
    RegionsOverlyPolymorphic(BoundRegion, Region),
    Sorts(ExpectedFound<Ty<'tcx>>),
    IntegerAsChar,
    IntMismatch(ExpectedFound<ty::IntVarValue>),
    FloatMismatch(ExpectedFound<ast::FloatTy>),
    Traits(ExpectedFound<DefId>),
    BuiltinBoundsMismatch(ExpectedFound<ty::BuiltinBounds>),
    VariadicMismatch(ExpectedFound<bool>),
    CyclicTy,
    ConvergenceMismatch(ExpectedFound<bool>),
    ProjectionNameMismatched(ExpectedFound<Name>),
    ProjectionBoundsLength(ExpectedFound<usize>),
    TyParamDefaultMismatch(ExpectedFound<type_variable::Default<'tcx>>)
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Eq, Hash, Debug, Copy)]
pub enum UnconstrainedNumeric {
    UnconstrainedFloat,
    UnconstrainedInt,
    Neither,
}

/// Explains the source of a type err in a short, human readable way. This is meant to be placed
/// in parentheses after some larger message. You should also invoke `note_and_explain_type_err()`
/// afterwards to present additional details, particularly when it comes to lifetime-related
/// errors.
impl<'tcx> fmt::Display for TypeError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::TypeError::*;
        fn report_maybe_different(f: &mut fmt::Formatter,
                                  expected: String, found: String) -> fmt::Result {
            // A naive approach to making sure that we're not reporting silly errors such as:
            // (expected closure, found closure).
            if expected == found {
                write!(f, "expected {}, found a different {}", expected, found)
            } else {
                write!(f, "expected {}, found {}", expected, found)
            }
        }

        match *self {
            CyclicTy => write!(f, "cyclic type of infinite size"),
            Mismatch => write!(f, "types differ"),
            UnsafetyMismatch(values) => {
                write!(f, "expected {} fn, found {} fn",
                       values.expected,
                       values.found)
            }
            AbiMismatch(values) => {
                write!(f, "expected {} fn, found {} fn",
                       values.expected,
                       values.found)
            }
            Mutability => write!(f, "values differ in mutability"),
            BoxMutability => {
                write!(f, "boxed values differ in mutability")
            }
            VecMutability => write!(f, "vectors differ in mutability"),
            PtrMutability => write!(f, "pointers differ in mutability"),
            RefMutability => write!(f, "references differ in mutability"),
            TyParamSize(values) => {
                write!(f, "expected a type with {} type params, \
                           found one with {} type params",
                       values.expected,
                       values.found)
            }
            FixedArraySize(values) => {
                write!(f, "expected an array with a fixed size of {} elements, \
                           found one with {} elements",
                       values.expected,
                       values.found)
            }
            TupleSize(values) => {
                write!(f, "expected a tuple with {} elements, \
                           found one with {} elements",
                       values.expected,
                       values.found)
            }
            ArgCount => {
                write!(f, "incorrect number of function parameters")
            }
            RegionsDoesNotOutlive(..) => {
                write!(f, "lifetime mismatch")
            }
            RegionsNotSame(..) => {
                write!(f, "lifetimes are not the same")
            }
            RegionsNoOverlap(..) => {
                write!(f, "lifetimes do not intersect")
            }
            RegionsInsufficientlyPolymorphic(br, _) => {
                write!(f, "expected bound lifetime parameter {}, \
                           found concrete lifetime", br)
            }
            RegionsOverlyPolymorphic(br, _) => {
                write!(f, "expected concrete lifetime, \
                           found bound lifetime parameter {}", br)
            }
            Sorts(values) => ty::tls::with(|tcx| {
                report_maybe_different(f, values.expected.sort_string(tcx),
                                       values.found.sort_string(tcx))
            }),
            Traits(values) => ty::tls::with(|tcx| {
                report_maybe_different(f,
                                       format!("trait `{}`",
                                               tcx.item_path_str(values.expected)),
                                       format!("trait `{}`",
                                               tcx.item_path_str(values.found)))
            }),
            BuiltinBoundsMismatch(values) => {
                if values.expected.is_empty() {
                    write!(f, "expected no bounds, found `{}`",
                           values.found)
                } else if values.found.is_empty() {
                    write!(f, "expected bounds `{}`, found no bounds",
                           values.expected)
                } else {
                    write!(f, "expected bounds `{}`, found bounds `{}`",
                           values.expected,
                           values.found)
                }
            }
            IntegerAsChar => {
                write!(f, "expected an integral type, found `char`")
            }
            IntMismatch(ref values) => {
                write!(f, "expected `{:?}`, found `{:?}`",
                       values.expected,
                       values.found)
            }
            FloatMismatch(ref values) => {
                write!(f, "expected `{:?}`, found `{:?}`",
                       values.expected,
                       values.found)
            }
            VariadicMismatch(ref values) => {
                write!(f, "expected {} fn, found {} function",
                       if values.expected { "variadic" } else { "non-variadic" },
                       if values.found { "variadic" } else { "non-variadic" })
            }
            ConvergenceMismatch(ref values) => {
                write!(f, "expected {} fn, found {} function",
                       if values.expected { "converging" } else { "diverging" },
                       if values.found { "converging" } else { "diverging" })
            }
            ProjectionNameMismatched(ref values) => {
                write!(f, "expected {}, found {}",
                       values.expected,
                       values.found)
            }
            ProjectionBoundsLength(ref values) => {
                write!(f, "expected {} associated type bindings, found {}",
                       values.expected,
                       values.found)
            },
            TyParamDefaultMismatch(ref values) => {
                write!(f, "conflicting type parameter defaults `{}` and `{}`",
                       values.expected.ty,
                       values.found.ty)
            }
        }
    }
}

impl<'tcx> ty::TyS<'tcx> {
    fn sort_string(&self, cx: &ty::ctxt) -> String {
        match self.sty {
            ty::TyBool | ty::TyChar | ty::TyInt(_) |
            ty::TyUint(_) | ty::TyFloat(_) | ty::TyStr => self.to_string(),
            ty::TyTuple(ref tys) if tys.is_empty() => self.to_string(),

            ty::TyEnum(def, _) => format!("enum `{}`", cx.item_path_str(def.did)),
            ty::TyBox(_) => "box".to_string(),
            ty::TyArray(_, n) => format!("array of {} elements", n),
            ty::TySlice(_) => "slice".to_string(),
            ty::TyRawPtr(_) => "*-ptr".to_string(),
            ty::TyRef(_, _) => "&-ptr".to_string(),
            ty::TyBareFn(Some(_), _) => format!("fn item"),
            ty::TyBareFn(None, _) => "fn pointer".to_string(),
            ty::TyTrait(ref inner) => {
                format!("trait {}", cx.item_path_str(inner.principal_def_id()))
            }
            ty::TyStruct(def, _) => {
                format!("struct `{}`", cx.item_path_str(def.did))
            }
            ty::TyClosure(..) => "closure".to_string(),
            ty::TyTuple(_) => "tuple".to_string(),
            ty::TyInfer(ty::TyVar(_)) => "inferred type".to_string(),
            ty::TyInfer(ty::IntVar(_)) => "integral variable".to_string(),
            ty::TyInfer(ty::FloatVar(_)) => "floating-point variable".to_string(),
            ty::TyInfer(ty::FreshTy(_)) => "skolemized type".to_string(),
            ty::TyInfer(ty::FreshIntTy(_)) => "skolemized integral type".to_string(),
            ty::TyInfer(ty::FreshFloatTy(_)) => "skolemized floating-point type".to_string(),
            ty::TyProjection(_) => "associated type".to_string(),
            ty::TyParam(ref p) => {
                if p.space == subst::SelfSpace {
                    "Self".to_string()
                } else {
                    "type parameter".to_string()
                }
            }
            ty::TyError => "type error".to_string(),
        }
    }
}

impl<'tcx> ty::ctxt<'tcx> {
    pub fn note_and_explain_type_err(&self, err: &TypeError<'tcx>, sp: Span) {
        use self::TypeError::*;

        match err.clone() {
            RegionsDoesNotOutlive(subregion, superregion) => {
                self.note_and_explain_region("", subregion, "...");
                self.note_and_explain_region("...does not necessarily outlive ",
                                           superregion, "");
            }
            RegionsNotSame(region1, region2) => {
                self.note_and_explain_region("", region1, "...");
                self.note_and_explain_region("...is not the same lifetime as ",
                                           region2, "");
            }
            RegionsNoOverlap(region1, region2) => {
                self.note_and_explain_region("", region1, "...");
                self.note_and_explain_region("...does not overlap ",
                                           region2, "");
            }
            RegionsInsufficientlyPolymorphic(_, conc_region) => {
                self.note_and_explain_region("concrete lifetime that was found is ",
                                           conc_region, "");
            }
            RegionsOverlyPolymorphic(_, ty::ReVar(_)) => {
                // don't bother to print out the message below for
                // inference variables, it's not very illuminating.
            }
            RegionsOverlyPolymorphic(_, conc_region) => {
                self.note_and_explain_region("expected concrete lifetime is ",
                                           conc_region, "");
            }
            Sorts(values) => {
                let expected_str = values.expected.sort_string(self);
                let found_str = values.found.sort_string(self);
                if expected_str == found_str && expected_str == "closure" {
                    self.sess.span_note(sp,
                        &format!("no two closures, even if identical, have the same type"));
                    self.sess.span_help(sp,
                        &format!("consider boxing your closure and/or \
                                  using it as a trait object"));
                }
            },
            TyParamDefaultMismatch(values) => {
                let expected = values.expected;
                let found = values.found;
                self.sess.span_note(sp,
                                    &format!("conflicting type parameter defaults `{}` and `{}`",
                                             expected.ty,
                                             found.ty));

                match
                    self.map.as_local_node_id(expected.def_id)
                            .and_then(|node_id| self.map.opt_span(node_id))
                {
                    Some(span) => {
                        self.sess.span_note(span,
                                            &format!("a default was defined here..."));
                    }
                    None => {
                        self.sess.note(
                            &format!("a default is defined on `{}`",
                                     self.item_path_str(expected.def_id)));
                    }
                }

                self.sess.span_note(
                    expected.origin_span,
                    &format!("...that was applied to an unconstrained type variable here"));

                match
                    self.map.as_local_node_id(found.def_id)
                            .and_then(|node_id| self.map.opt_span(node_id))
                {
                    Some(span) => {
                        self.sess.span_note(span,
                                            &format!("a second default was defined here..."));
                    }
                    None => {
                        self.sess.note(
                            &format!("a second default is defined on `{}`",
                                     self.item_path_str(found.def_id)));
                    }
                }

                self.sess.span_note(
                    found.origin_span,
                    &format!("...that also applies to the same type variable here"));
            }
            _ => {}
        }
    }
}
