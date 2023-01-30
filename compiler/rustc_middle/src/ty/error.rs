use crate::ty::print::{with_forced_trimmed_paths, FmtPrinter, PrettyPrinter};
use crate::ty::{self, BoundRegionKind, Region, Ty, TyCtxt};
use rustc_errors::pluralize;
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind};
use rustc_hir::def_id::DefId;
use rustc_span::symbol::Symbol;
use rustc_target::spec::abi;
use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, TypeFoldable, TypeVisitable, Lift)]
pub struct ExpectedFound<T> {
    pub expected: T,
    pub found: T,
}

impl<T> ExpectedFound<T> {
    pub fn new(a_is_expected: bool, a: T, b: T) -> Self {
        if a_is_expected {
            ExpectedFound { expected: a, found: b }
        } else {
            ExpectedFound { expected: b, found: a }
        }
    }
}

// Data structures used in type unification
#[derive(Copy, Clone, Debug, TypeFoldable, TypeVisitable, Lift)]
#[rustc_pass_by_value]
pub enum TypeError<'tcx> {
    Mismatch,
    ConstnessMismatch(ExpectedFound<ty::BoundConstness>),
    PolarityMismatch(ExpectedFound<ty::ImplPolarity>),
    UnsafetyMismatch(ExpectedFound<hir::Unsafety>),
    AbiMismatch(ExpectedFound<abi::Abi>),
    Mutability,
    ArgumentMutability(usize),
    TupleSize(ExpectedFound<usize>),
    FixedArraySize(ExpectedFound<u64>),
    ArgCount,
    FieldMisMatch(Symbol, Symbol),

    RegionsDoesNotOutlive(Region<'tcx>, Region<'tcx>),
    RegionsInsufficientlyPolymorphic(BoundRegionKind, Region<'tcx>),
    RegionsOverlyPolymorphic(BoundRegionKind, Region<'tcx>),
    RegionsPlaceholderMismatch,

    Sorts(ExpectedFound<Ty<'tcx>>),
    ArgumentSorts(ExpectedFound<Ty<'tcx>>, usize),
    IntMismatch(ExpectedFound<ty::IntVarValue>),
    FloatMismatch(ExpectedFound<ty::FloatTy>),
    Traits(ExpectedFound<DefId>),
    VariadicMismatch(ExpectedFound<bool>),

    /// Instantiating a type variable with the given type would have
    /// created a cycle (because it appears somewhere within that
    /// type).
    CyclicTy(Ty<'tcx>),
    CyclicConst(ty::Const<'tcx>),
    ProjectionMismatched(ExpectedFound<DefId>),
    ExistentialMismatch(ExpectedFound<&'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>>),
    ConstMismatch(ExpectedFound<ty::Const<'tcx>>),

    IntrinsicCast,
    /// Safe `#[target_feature]` functions are not assignable to safe function pointers.
    TargetFeatureCast(DefId),
}

impl TypeError<'_> {
    pub fn involves_regions(self) -> bool {
        match self {
            TypeError::RegionsDoesNotOutlive(_, _)
            | TypeError::RegionsInsufficientlyPolymorphic(_, _)
            | TypeError::RegionsOverlyPolymorphic(_, _)
            | TypeError::RegionsPlaceholderMismatch => true,
            _ => false,
        }
    }
}

/// Explains the source of a type err in a short, human readable way. This is meant to be placed
/// in parentheses after some larger message. You should also invoke `note_and_explain_type_err()`
/// afterwards to present additional details, particularly when it comes to lifetime-related
/// errors.
impl<'tcx> fmt::Display for TypeError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::TypeError::*;
        fn report_maybe_different(
            f: &mut fmt::Formatter<'_>,
            expected: &str,
            found: &str,
        ) -> fmt::Result {
            // A naive approach to making sure that we're not reporting silly errors such as:
            // (expected closure, found closure).
            if expected == found {
                write!(f, "expected {}, found a different {}", expected, found)
            } else {
                write!(f, "expected {}, found {}", expected, found)
            }
        }

        let br_string = |br: ty::BoundRegionKind| match br {
            ty::BrNamed(_, name) => format!(" {}", name),
            _ => String::new(),
        };

        match *self {
            CyclicTy(_) => write!(f, "cyclic type of infinite size"),
            CyclicConst(_) => write!(f, "encountered a self-referencing constant"),
            Mismatch => write!(f, "types differ"),
            ConstnessMismatch(values) => {
                write!(f, "expected {} bound, found {} bound", values.expected, values.found)
            }
            PolarityMismatch(values) => {
                write!(f, "expected {} polarity, found {} polarity", values.expected, values.found)
            }
            UnsafetyMismatch(values) => {
                write!(f, "expected {} fn, found {} fn", values.expected, values.found)
            }
            AbiMismatch(values) => {
                write!(f, "expected {} fn, found {} fn", values.expected, values.found)
            }
            ArgumentMutability(_) | Mutability => write!(f, "types differ in mutability"),
            TupleSize(values) => write!(
                f,
                "expected a tuple with {} element{}, found one with {} element{}",
                values.expected,
                pluralize!(values.expected),
                values.found,
                pluralize!(values.found)
            ),
            FixedArraySize(values) => write!(
                f,
                "expected an array with a fixed size of {} element{}, found one with {} element{}",
                values.expected,
                pluralize!(values.expected),
                values.found,
                pluralize!(values.found)
            ),
            ArgCount => write!(f, "incorrect number of function parameters"),
            FieldMisMatch(adt, field) => write!(f, "field type mismatch: {}.{}", adt, field),
            RegionsDoesNotOutlive(..) => write!(f, "lifetime mismatch"),
            // Actually naming the region here is a bit confusing because context is lacking
            RegionsInsufficientlyPolymorphic(..) => {
                write!(f, "one type is more general than the other")
            }
            RegionsOverlyPolymorphic(br, _) => write!(
                f,
                "expected concrete lifetime, found bound lifetime parameter{}",
                br_string(br)
            ),
            RegionsPlaceholderMismatch => write!(f, "one type is more general than the other"),
            ArgumentSorts(values, _) | Sorts(values) => ty::tls::with(|tcx| {
                let (mut expected, mut found) = with_forced_trimmed_paths!((
                    values.expected.sort_string(tcx),
                    values.found.sort_string(tcx),
                ));
                if expected == found {
                    expected = values.expected.sort_string(tcx);
                    found = values.found.sort_string(tcx);
                }
                report_maybe_different(f, &expected, &found)
            }),
            Traits(values) => ty::tls::with(|tcx| {
                let (mut expected, mut found) = with_forced_trimmed_paths!((
                    tcx.def_path_str(values.expected),
                    tcx.def_path_str(values.found),
                ));
                if expected == found {
                    expected = tcx.def_path_str(values.expected);
                    found = tcx.def_path_str(values.found);
                }
                report_maybe_different(
                    f,
                    &format!("trait `{expected}`"),
                    &format!("trait `{found}`"),
                )
            }),
            IntMismatch(ref values) => {
                let expected = match values.expected {
                    ty::IntVarValue::IntType(ty) => ty.name_str(),
                    ty::IntVarValue::UintType(ty) => ty.name_str(),
                };
                let found = match values.found {
                    ty::IntVarValue::IntType(ty) => ty.name_str(),
                    ty::IntVarValue::UintType(ty) => ty.name_str(),
                };
                write!(f, "expected `{}`, found `{}`", expected, found)
            }
            FloatMismatch(ref values) => {
                write!(
                    f,
                    "expected `{}`, found `{}`",
                    values.expected.name_str(),
                    values.found.name_str()
                )
            }
            VariadicMismatch(ref values) => write!(
                f,
                "expected {} fn, found {} function",
                if values.expected { "variadic" } else { "non-variadic" },
                if values.found { "variadic" } else { "non-variadic" }
            ),
            ProjectionMismatched(ref values) => ty::tls::with(|tcx| {
                write!(
                    f,
                    "expected {}, found {}",
                    tcx.def_path_str(values.expected),
                    tcx.def_path_str(values.found)
                )
            }),
            ExistentialMismatch(ref values) => report_maybe_different(
                f,
                &format!("trait `{}`", values.expected),
                &format!("trait `{}`", values.found),
            ),
            ConstMismatch(ref values) => {
                write!(f, "expected `{}`, found `{}`", values.expected, values.found)
            }
            IntrinsicCast => write!(f, "cannot coerce intrinsics to function pointers"),
            TargetFeatureCast(_) => write!(
                f,
                "cannot coerce functions with `#[target_feature]` to safe function pointers"
            ),
        }
    }
}

impl<'tcx> TypeError<'tcx> {
    pub fn must_include_note(self) -> bool {
        use self::TypeError::*;
        match self {
            CyclicTy(_) | CyclicConst(_) | UnsafetyMismatch(_) | ConstnessMismatch(_)
            | PolarityMismatch(_) | Mismatch | AbiMismatch(_) | FixedArraySize(_)
            | ArgumentSorts(..) | Sorts(_) | IntMismatch(_) | FloatMismatch(_)
            | VariadicMismatch(_) | TargetFeatureCast(_) => false,

            Mutability
            | ArgumentMutability(_)
            | TupleSize(_)
            | ArgCount
            | FieldMisMatch(..)
            | RegionsDoesNotOutlive(..)
            | RegionsInsufficientlyPolymorphic(..)
            | RegionsOverlyPolymorphic(..)
            | RegionsPlaceholderMismatch
            | Traits(_)
            | ProjectionMismatched(_)
            | ExistentialMismatch(_)
            | ConstMismatch(_)
            | IntrinsicCast => true,
        }
    }
}

impl<'tcx> Ty<'tcx> {
    pub fn sort_string(self, tcx: TyCtxt<'_>) -> Cow<'static, str> {
        match *self.kind() {
            ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Str | ty::Never => {
                format!("`{}`", self).into()
            }
            ty::Tuple(ref tys) if tys.is_empty() => format!("`{}`", self).into(),

            ty::Adt(def, _) => format!("{} `{}`", def.descr(), tcx.def_path_str(def.did())).into(),
            ty::Foreign(def_id) => format!("extern type `{}`", tcx.def_path_str(def_id)).into(),
            ty::Array(t, n) => {
                if t.is_simple_ty() {
                    return format!("array `{}`", self).into();
                }

                let n = tcx.lift(n).unwrap();
                if let ty::ConstKind::Value(v) = n.kind() {
                    if let Some(n) = v.try_to_machine_usize(tcx) {
                        return format!("array of {} element{}", n, pluralize!(n)).into();
                    }
                }
                "array".into()
            }
            ty::Slice(ty) if ty.is_simple_ty() => format!("slice `{}`", self).into(),
            ty::Slice(_) => "slice".into(),
            ty::RawPtr(tymut) => {
                let tymut_string = match tymut.mutbl {
                    hir::Mutability::Mut => tymut.to_string(),
                    hir::Mutability::Not => format!("const {}", tymut.ty),
                };

                if tymut_string != "_" && (tymut.ty.is_simple_text() || tymut_string.len() < "const raw pointer".len()) {
                    format!("`*{}`", tymut_string).into()
                } else {
                    // Unknown type name, it's long or has type arguments
                    "raw pointer".into()
                }
            },
            ty::Ref(_, ty, mutbl) => {
                let tymut = ty::TypeAndMut { ty, mutbl };
                let tymut_string = tymut.to_string();

                if tymut_string != "_"
                    && (ty.is_simple_text() || tymut_string.len() < "mutable reference".len())
                {
                    format!("`&{}`", tymut_string).into()
                } else {
                    // Unknown type name, it's long or has type arguments
                    match mutbl {
                        hir::Mutability::Mut => "mutable reference",
                        _ => "reference",
                    }
                    .into()
                }
            }
            ty::FnDef(def_id, ..) => match tcx.def_kind(def_id) {
                DefKind::Ctor(CtorOf::Struct, _) => "struct constructor".into(),
                DefKind::Ctor(CtorOf::Variant, _) => "enum constructor".into(),
                _ => "fn item".into(),
            },
            ty::FnPtr(_) => "fn pointer".into(),
            ty::Dynamic(ref inner, ..) if let Some(principal) = inner.principal() => {
                format!("trait object `dyn {}`", tcx.def_path_str(principal.def_id())).into()
            }
            ty::Dynamic(..) => "trait object".into(),
            ty::Closure(..) => "closure".into(),
            ty::Generator(def_id, ..) => tcx.generator_kind(def_id).unwrap().descr().into(),
            ty::GeneratorWitness(..) |
            ty::GeneratorWitnessMIR(..) => "generator witness".into(),
            ty::Tuple(..) => "tuple".into(),
            ty::Infer(ty::TyVar(_)) => "inferred type".into(),
            ty::Infer(ty::IntVar(_)) => "integer".into(),
            ty::Infer(ty::FloatVar(_)) => "floating-point number".into(),
            ty::Placeholder(..) => "placeholder type".into(),
            ty::Bound(..) => "bound type".into(),
            ty::Infer(ty::FreshTy(_)) => "fresh type".into(),
            ty::Infer(ty::FreshIntTy(_)) => "fresh integral type".into(),
            ty::Infer(ty::FreshFloatTy(_)) => "fresh floating-point type".into(),
            ty::Alias(ty::Projection, _) => "associated type".into(),
            ty::Param(p) => format!("type parameter `{}`", p).into(),
            ty::Alias(ty::Opaque, ..) => "opaque type".into(),
            ty::Error(_) => "type error".into(),
        }
    }

    pub fn prefix_string(self, tcx: TyCtxt<'_>) -> Cow<'static, str> {
        match *self.kind() {
            ty::Infer(_)
            | ty::Error(_)
            | ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Never => "type".into(),
            ty::Tuple(ref tys) if tys.is_empty() => "unit type".into(),
            ty::Adt(def, _) => def.descr().into(),
            ty::Foreign(_) => "extern type".into(),
            ty::Array(..) => "array".into(),
            ty::Slice(_) => "slice".into(),
            ty::RawPtr(_) => "raw pointer".into(),
            ty::Ref(.., mutbl) => match mutbl {
                hir::Mutability::Mut => "mutable reference",
                _ => "reference",
            }
            .into(),
            ty::FnDef(def_id, ..) => match tcx.def_kind(def_id) {
                DefKind::Ctor(CtorOf::Struct, _) => "struct constructor".into(),
                DefKind::Ctor(CtorOf::Variant, _) => "enum constructor".into(),
                _ => "fn item".into(),
            },
            ty::FnPtr(_) => "fn pointer".into(),
            ty::Dynamic(..) => "trait object".into(),
            ty::Closure(..) => "closure".into(),
            ty::Generator(def_id, ..) => tcx.generator_kind(def_id).unwrap().descr().into(),
            ty::GeneratorWitness(..) | ty::GeneratorWitnessMIR(..) => "generator witness".into(),
            ty::Tuple(..) => "tuple".into(),
            ty::Placeholder(..) => "higher-ranked type".into(),
            ty::Bound(..) => "bound type variable".into(),
            ty::Alias(ty::Projection, _) => "associated type".into(),
            ty::Param(_) => "type parameter".into(),
            ty::Alias(ty::Opaque, ..) => "opaque type".into(),
        }
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn short_ty_string(self, ty: Ty<'tcx>) -> (String, Option<PathBuf>) {
        let width = self.sess.diagnostic_width();
        let length_limit = width.saturating_sub(30);
        let mut type_limit = 50;
        let regular = FmtPrinter::new(self, hir::def::Namespace::TypeNS)
            .pretty_print_type(ty)
            .expect("could not write to `String`")
            .into_buffer();
        if regular.len() <= width {
            return (regular, None);
        }
        let mut short;
        loop {
            // Look for the longest properly trimmed path that still fits in length_limit.
            short = with_forced_trimmed_paths!(
                FmtPrinter::new_with_limit(
                    self,
                    hir::def::Namespace::TypeNS,
                    rustc_session::Limit(type_limit),
                )
                .pretty_print_type(ty)
                .expect("could not write to `String`")
                .into_buffer()
            );
            if short.len() <= length_limit || type_limit == 0 {
                break;
            }
            type_limit -= 1;
        }
        if regular == short {
            return (regular, None);
        }
        // Multiple types might be shortened in a single error, ensure we create a file for each.
        let mut s = DefaultHasher::new();
        ty.hash(&mut s);
        let hash = s.finish();
        let path = self.output_filenames(()).temp_path_ext(&format!("long-type-{hash}.txt"), None);
        match std::fs::write(&path, &regular) {
            Ok(_) => (short, Some(path)),
            Err(_) => (regular, None),
        }
    }
}
