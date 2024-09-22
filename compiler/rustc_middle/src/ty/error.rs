use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;

use rustc_errors::pluralize;
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind};
use rustc_macros::extension;
pub use rustc_type_ir::error::ExpectedFound;

use crate::ty::print::{FmtPrinter, PrettyPrinter, with_forced_trimmed_paths};
use crate::ty::{self, Ty, TyCtxt};

pub type TypeError<'tcx> = rustc_type_ir::error::TypeError<TyCtxt<'tcx>>;

/// Explains the source of a type err in a short, human readable way.
/// This is meant to be placed in parentheses after some larger message.
/// You should also invoke `note_and_explain_type_err()` afterwards
/// to present additional details, particularly when it comes to lifetime-
/// related errors.
#[extension(pub trait TypeErrorToStringExt<'tcx>)]
impl<'tcx> TypeError<'tcx> {
    fn to_string(self, tcx: TyCtxt<'tcx>) -> Cow<'static, str> {
        fn report_maybe_different(expected: &str, found: &str) -> String {
            // A naive approach to making sure that we're not reporting silly errors such as:
            // (expected closure, found closure).
            if expected == found {
                format!("expected {expected}, found a different {found}")
            } else {
                format!("expected {expected}, found {found}")
            }
        }

        match self {
            TypeError::CyclicTy(_) => "cyclic type of infinite size".into(),
            TypeError::CyclicConst(_) => "encountered a self-referencing constant".into(),
            TypeError::Mismatch => "types differ".into(),
            TypeError::ConstnessMismatch(values) => {
                format!("expected {} bound, found {} bound", values.expected, values.found).into()
            }
            TypeError::PolarityMismatch(values) => {
                format!("expected {} polarity, found {} polarity", values.expected, values.found)
                    .into()
            }
            TypeError::SafetyMismatch(values) => {
                format!("expected {} fn, found {} fn", values.expected, values.found).into()
            }
            TypeError::AbiMismatch(values) => {
                format!("expected {} fn, found {} fn", values.expected, values.found).into()
            }
            TypeError::ArgumentMutability(_) | TypeError::Mutability => {
                "types differ in mutability".into()
            }
            TypeError::TupleSize(values) => format!(
                "expected a tuple with {} element{}, found one with {} element{}",
                values.expected,
                pluralize!(values.expected),
                values.found,
                pluralize!(values.found)
            )
            .into(),
            TypeError::FixedArraySize(values) => format!(
                "expected an array with a fixed size of {} element{}, found one with {} element{}",
                values.expected,
                pluralize!(values.expected),
                values.found,
                pluralize!(values.found)
            )
            .into(),
            TypeError::ArgCount => "incorrect number of function parameters".into(),
            TypeError::RegionsDoesNotOutlive(..) => "lifetime mismatch".into(),
            // Actually naming the region here is a bit confusing because context is lacking
            TypeError::RegionsInsufficientlyPolymorphic(..) => {
                "one type is more general than the other".into()
            }
            TypeError::RegionsPlaceholderMismatch => {
                "one type is more general than the other".into()
            }
            TypeError::ArgumentSorts(values, _) | TypeError::Sorts(values) => {
                let expected = values.expected.sort_string(tcx);
                let found = values.found.sort_string(tcx);
                report_maybe_different(&expected, &found).into()
            }
            TypeError::Traits(values) => {
                let (mut expected, mut found) = with_forced_trimmed_paths!((
                    tcx.def_path_str(values.expected),
                    tcx.def_path_str(values.found),
                ));
                if expected == found {
                    expected = tcx.def_path_str(values.expected);
                    found = tcx.def_path_str(values.found);
                }
                report_maybe_different(&format!("trait `{expected}`"), &format!("trait `{found}`"))
                    .into()
            }
            TypeError::VariadicMismatch(ref values) => format!(
                "expected {} fn, found {} function",
                if values.expected { "variadic" } else { "non-variadic" },
                if values.found { "variadic" } else { "non-variadic" }
            )
            .into(),
            TypeError::ProjectionMismatched(ref values) => format!(
                "expected `{}`, found `{}`",
                tcx.def_path_str(values.expected),
                tcx.def_path_str(values.found)
            )
            .into(),
            TypeError::ExistentialMismatch(ref values) => report_maybe_different(
                &format!("trait `{}`", values.expected),
                &format!("trait `{}`", values.found),
            )
            .into(),
            TypeError::ConstMismatch(ref values) => {
                format!("expected `{}`, found `{}`", values.expected, values.found).into()
            }
            TypeError::IntrinsicCast => "cannot coerce intrinsics to function pointers".into(),
            TypeError::TargetFeatureCast(_) => {
                "cannot coerce functions with `#[target_feature]` to safe function pointers".into()
            }
        }
    }
}

impl<'tcx> Ty<'tcx> {
    pub fn sort_string(self, tcx: TyCtxt<'tcx>) -> Cow<'static, str> {
        match *self.kind() {
            ty::Foreign(def_id) => format!("extern type `{}`", tcx.def_path_str(def_id)).into(),
            ty::FnDef(def_id, ..) => match tcx.def_kind(def_id) {
                DefKind::Ctor(CtorOf::Struct, _) => "struct constructor".into(),
                DefKind::Ctor(CtorOf::Variant, _) => "enum constructor".into(),
                _ => "fn item".into(),
            },
            ty::FnPtr(..) => "fn pointer".into(),
            ty::Dynamic(inner, ..) if let Some(principal) = inner.principal() => {
                format!("`dyn {}`", tcx.def_path_str(principal.def_id())).into()
            }
            ty::Dynamic(..) => "trait object".into(),
            ty::Closure(..) => "closure".into(),
            ty::Coroutine(def_id, ..) => {
                format!("{:#}", tcx.coroutine_kind(def_id).unwrap()).into()
            }
            ty::CoroutineWitness(..) => "coroutine witness".into(),
            ty::Infer(ty::TyVar(_)) => "inferred type".into(),
            ty::Infer(ty::IntVar(_)) => "integer".into(),
            ty::Infer(ty::FloatVar(_)) => "floating-point number".into(),
            ty::Placeholder(..) => "placeholder type".into(),
            ty::Bound(..) => "bound type".into(),
            ty::Infer(ty::FreshTy(_)) => "fresh type".into(),
            ty::Infer(ty::FreshIntTy(_)) => "fresh integral type".into(),
            ty::Infer(ty::FreshFloatTy(_)) => "fresh floating-point type".into(),
            ty::Alias(ty::Projection | ty::Inherent, _) => "associated type".into(),
            ty::Param(p) => format!("type parameter `{p}`").into(),
            ty::Alias(ty::Opaque, ..) => {
                if tcx.ty_is_opaque_future(self) {
                    "future".into()
                } else {
                    "opaque type".into()
                }
            }
            ty::Error(_) => "type error".into(),
            _ => {
                let width = tcx.sess.diagnostic_width();
                let length_limit = std::cmp::max(width / 4, 15);
                format!("`{}`", tcx.ty_string_with_limit(self, length_limit)).into()
            }
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
            ty::Tuple(tys) if tys.is_empty() => "unit type".into(),
            ty::Adt(def, _) => def.descr().into(),
            ty::Foreign(_) => "extern type".into(),
            ty::Array(..) => "array".into(),
            ty::Pat(..) => "pattern type".into(),
            ty::Slice(_) => "slice".into(),
            ty::RawPtr(_, _) => "raw pointer".into(),
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
            ty::FnPtr(..) => "fn pointer".into(),
            ty::Dynamic(..) => "trait object".into(),
            ty::Closure(..) | ty::CoroutineClosure(..) => "closure".into(),
            ty::Coroutine(def_id, ..) => {
                format!("{:#}", tcx.coroutine_kind(def_id).unwrap()).into()
            }
            ty::CoroutineWitness(..) => "coroutine witness".into(),
            ty::Tuple(..) => "tuple".into(),
            ty::Placeholder(..) => "higher-ranked type".into(),
            ty::Bound(..) => "bound type variable".into(),
            ty::Alias(ty::Projection | ty::Inherent, _) => "associated type".into(),
            ty::Alias(ty::Weak, _) => "type alias".into(),
            ty::Param(_) => "type parameter".into(),
            ty::Alias(ty::Opaque, ..) => "opaque type".into(),
        }
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn ty_string_with_limit(self, ty: Ty<'tcx>, length_limit: usize) -> String {
        let mut type_limit = 50;
        let regular = FmtPrinter::print_string(self, hir::def::Namespace::TypeNS, |cx| {
            cx.pretty_print_type(ty)
        })
        .expect("could not write to `String`");
        if regular.len() <= length_limit {
            return regular;
        }
        let mut short;
        loop {
            // Look for the longest properly trimmed path that still fits in length_limit.
            short = with_forced_trimmed_paths!({
                let mut cx = FmtPrinter::new_with_limit(
                    self,
                    hir::def::Namespace::TypeNS,
                    rustc_session::Limit(type_limit),
                );
                cx.pretty_print_type(ty).expect("could not write to `String`");
                cx.into_buffer()
            });
            if short.len() <= length_limit || type_limit == 0 {
                break;
            }
            type_limit -= 1;
        }
        short
    }

    pub fn short_ty_string(self, ty: Ty<'tcx>, path: &mut Option<PathBuf>) -> String {
        let regular = FmtPrinter::print_string(self, hir::def::Namespace::TypeNS, |cx| {
            cx.pretty_print_type(ty)
        })
        .expect("could not write to `String`");

        if !self.sess.opts.unstable_opts.write_long_types_to_disk || self.sess.opts.verbose {
            return regular;
        }

        let width = self.sess.diagnostic_width();
        let length_limit = width.saturating_sub(30);
        if regular.len() <= width {
            return regular;
        }
        let short = self.ty_string_with_limit(ty, length_limit);
        if regular == short {
            return regular;
        }
        // Ensure we create an unique file for the type passed in when we create a file.
        let mut s = DefaultHasher::new();
        ty.hash(&mut s);
        let hash = s.finish();
        *path = Some(path.take().unwrap_or_else(|| {
            self.output_filenames(()).temp_path_ext(&format!("long-type-{hash}.txt"), None)
        }));
        match std::fs::write(path.as_ref().unwrap(), &format!("{regular}\n")) {
            Ok(_) => short,
            Err(_) => regular,
        }
    }
}
