use crate::hir::def_id::DefId;
use crate::ty::{self, BoundRegion, Region, Ty, TyCtxt};
use std::borrow::Cow;
use std::fmt;
use rustc_target::spec::abi;
use syntax::ast;
use errors::{Applicability, DiagnosticBuilder};
use syntax_pos::Span;

use crate::hir;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpectedFound<T> {
    pub expected: T,
    pub found: T,
}

// Data structures used in type unification
#[derive(Clone, Debug)]
pub enum TypeError<'tcx> {
    Mismatch,
    UnsafetyMismatch(ExpectedFound<hir::Unsafety>),
    AbiMismatch(ExpectedFound<abi::Abi>),
    Mutability,
    TupleSize(ExpectedFound<usize>),
    FixedArraySize(ExpectedFound<u64>),
    ArgCount,

    RegionsDoesNotOutlive(Region<'tcx>, Region<'tcx>),
    RegionsInsufficientlyPolymorphic(BoundRegion, Region<'tcx>),
    RegionsOverlyPolymorphic(BoundRegion, Region<'tcx>),
    RegionsPlaceholderMismatch,

    Sorts(ExpectedFound<Ty<'tcx>>),
    IntMismatch(ExpectedFound<ty::IntVarValue>),
    FloatMismatch(ExpectedFound<ast::FloatTy>),
    Traits(ExpectedFound<DefId>),
    VariadicMismatch(ExpectedFound<bool>),

    /// Instantiating a type variable with the given type would have
    /// created a cycle (because it appears somewhere within that
    /// type).
    CyclicTy(Ty<'tcx>),
    ProjectionMismatched(ExpectedFound<DefId>),
    ProjectionBoundsLength(ExpectedFound<usize>),
    ExistentialMismatch(ExpectedFound<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>>),

    ConstMismatch(ExpectedFound<&'tcx ty::Const<'tcx>>),
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::TypeError::*;
        fn report_maybe_different(f: &mut fmt::Formatter<'_>,
                                  expected: &str, found: &str) -> fmt::Result {
            // A naive approach to making sure that we're not reporting silly errors such as:
            // (expected closure, found closure).
            if expected == found {
                write!(f, "expected {}, found a different {}", expected, found)
            } else {
                write!(f, "expected {}, found {}", expected, found)
            }
        }

        let br_string = |br: ty::BoundRegion| {
            match br {
                ty::BrNamed(_, name) => format!(" {}", name),
                _ => String::new(),
            }
        };

        macro_rules! pluralise {
            ($x:expr) => {
                if $x != 1 { "s" } else { "" }
            };
        }

        match *self {
            CyclicTy(_) => write!(f, "cyclic type of infinite size"),
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
            Mutability => write!(f, "types differ in mutability"),
            TupleSize(values) => {
                write!(f, "expected a tuple with {} element{}, \
                           found one with {} element{}",
                       values.expected,
                       pluralise!(values.expected),
                       values.found,
                       pluralise!(values.found))
            }
            FixedArraySize(values) => {
                write!(f, "expected an array with a fixed size of {} element{}, \
                           found one with {} element{}",
                       values.expected,
                       pluralise!(values.expected),
                       values.found,
                       pluralise!(values.found))
            }
            ArgCount => {
                write!(f, "incorrect number of function parameters")
            }
            RegionsDoesNotOutlive(..) => {
                write!(f, "lifetime mismatch")
            }
            RegionsInsufficientlyPolymorphic(br, _) => {
                write!(f,
                       "expected bound lifetime parameter{}, found concrete lifetime",
                       br_string(br))
            }
            RegionsOverlyPolymorphic(br, _) => {
                write!(f,
                       "expected concrete lifetime, found bound lifetime parameter{}",
                       br_string(br))
            }
            RegionsPlaceholderMismatch => {
                write!(f, "one type is more general than the other")
            }
            Sorts(values) => ty::tls::with(|tcx| {
                report_maybe_different(f, &values.expected.sort_string(tcx),
                                       &values.found.sort_string(tcx))
            }),
            Traits(values) => ty::tls::with(|tcx| {
                report_maybe_different(f,
                                       &format!("trait `{}`",
                                                tcx.def_path_str(values.expected)),
                                       &format!("trait `{}`",
                                                tcx.def_path_str(values.found)))
            }),
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
            ProjectionMismatched(ref values) => ty::tls::with(|tcx| {
                write!(f, "expected {}, found {}",
                       tcx.def_path_str(values.expected),
                       tcx.def_path_str(values.found))
            }),
            ProjectionBoundsLength(ref values) => {
                write!(f, "expected {} associated type binding{}, found {}",
                       values.expected,
                       pluralise!(values.expected),
                       values.found)
            },
            ExistentialMismatch(ref values) => {
                report_maybe_different(f, &format!("trait `{}`", values.expected),
                                       &format!("trait `{}`", values.found))
            }
            ConstMismatch(ref values) => {
                write!(f, "expected `{}`, found `{}`", values.expected, values.found)
            }
        }
    }
}

impl<'tcx> ty::TyS<'tcx> {
    pub fn sort_string(&self, tcx: TyCtxt<'_>) -> Cow<'static, str> {
        match self.sty {
            ty::Bool | ty::Char | ty::Int(_) |
            ty::Uint(_) | ty::Float(_) | ty::Str | ty::Never => self.to_string().into(),
            ty::Tuple(ref tys) if tys.is_empty() => self.to_string().into(),

            ty::Adt(def, _) => format!("{} `{}`", def.descr(), tcx.def_path_str(def.did)).into(),
            ty::Foreign(def_id) => format!("extern type `{}`", tcx.def_path_str(def_id)).into(),
            ty::Array(_, n) => {
                let n = tcx.lift_to_global(&n).unwrap();
                match n.assert_usize(tcx) {
                    Some(n) => format!("array of {} elements", n).into(),
                    None => "array".into(),
                }
            }
            ty::Slice(_) => "slice".into(),
            ty::RawPtr(_) => "*-ptr".into(),
            ty::Ref(region, ty, mutbl) => {
                let tymut = ty::TypeAndMut { ty, mutbl };
                let tymut_string = tymut.to_string();
                if tymut_string == "_" ||         //unknown type name,
                   tymut_string.len() > 10 ||     //name longer than saying "reference",
                   region.to_string() != "'_"     //... or a complex type
                {
                    format!("{}reference", match mutbl {
                        hir::Mutability::MutMutable => "mutable ",
                        _ => ""
                    }).into()
                } else {
                    format!("&{}", tymut_string).into()
                }
            }
            ty::FnDef(..) => "fn item".into(),
            ty::FnPtr(_) => "fn pointer".into(),
            ty::Dynamic(ref inner, ..) => {
                if let Some(principal) = inner.principal() {
                    format!("trait {}", tcx.def_path_str(principal.def_id())).into()
                } else {
                    "trait".into()
                }
            }
            ty::Closure(..) => "closure".into(),
            ty::Generator(..) => "generator".into(),
            ty::GeneratorWitness(..) => "generator witness".into(),
            ty::Tuple(..) => "tuple".into(),
            ty::Infer(ty::TyVar(_)) => "inferred type".into(),
            ty::Infer(ty::IntVar(_)) => "integer".into(),
            ty::Infer(ty::FloatVar(_)) => "floating-point number".into(),
            ty::Placeholder(..) => "placeholder type".into(),
            ty::Bound(..) => "bound type".into(),
            ty::Infer(ty::FreshTy(_)) => "fresh type".into(),
            ty::Infer(ty::FreshIntTy(_)) => "fresh integral type".into(),
            ty::Infer(ty::FreshFloatTy(_)) => "fresh floating-point type".into(),
            ty::Projection(_) => "associated type".into(),
            ty::UnnormalizedProjection(_) => "non-normalized associated type".into(),
            ty::Param(ref p) => {
                if p.is_self() {
                    "Self".into()
                } else {
                    "type parameter".into()
                }
            }
            ty::Opaque(..) => "opaque type".into(),
            ty::Error => "type error".into(),
        }
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn note_and_explain_type_err(self,
                                     db: &mut DiagnosticBuilder<'_>,
                                     err: &TypeError<'tcx>,
                                     sp: Span) {
        use self::TypeError::*;

        match err.clone() {
            Sorts(values) => {
                let expected_str = values.expected.sort_string(self);
                let found_str = values.found.sort_string(self);
                if expected_str == found_str && expected_str == "closure" {
                    db.note("no two closures, even if identical, have the same type");
                    db.help("consider boxing your closure and/or using it as a trait object");
                }
                if let (ty::Infer(ty::IntVar(_)), ty::Float(_)) =
                       (&values.found.sty, &values.expected.sty) // Issue #53280
                {
                    if let Ok(snippet) = self.sess.source_map().span_to_snippet(sp) {
                        if snippet.chars().all(|c| c.is_digit(10) || c == '-' || c == '_') {
                            db.span_suggestion(
                                sp,
                                "use a float literal",
                                format!("{}.0", snippet),
                                Applicability::MachineApplicable
                            );
                        }
                    }
                }
            },
            CyclicTy(ty) => {
                // Watch out for various cases of cyclic types and try to explain.
                if ty.is_closure() || ty.is_generator() {
                    db.note("closures cannot capture themselves or take themselves as argument;\n\
                             this error may be the result of a recent compiler bug-fix,\n\
                             see https://github.com/rust-lang/rust/issues/46062 for more details");
                }
            }
            _ => {}
        }
    }
}
