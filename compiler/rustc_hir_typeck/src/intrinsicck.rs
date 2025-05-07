use hir::HirId;
use rustc_abi::Primitive::Pointer;
use rustc_abi::VariantIdx;
use rustc_errors::codes::*;
use rustc_errors::struct_span_code_err;
use rustc_hir as hir;
use rustc_index::Idx;
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutError, SizeSkeleton};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};
use tracing::trace;

use super::FnCtxt;

/// If the type is `Option<T>`, it will return `T`, otherwise
/// the type itself. Works on most `Option`-like types.
fn unpack_option_like<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    let ty::Adt(def, args) = *ty.kind() else { return ty };

    if def.variants().len() == 2 && !def.repr().c() && def.repr().int.is_none() {
        let data_idx;

        let one = VariantIdx::new(1);
        let zero = VariantIdx::ZERO;

        if def.variant(zero).fields.is_empty() {
            data_idx = one;
        } else if def.variant(one).fields.is_empty() {
            data_idx = zero;
        } else {
            return ty;
        }

        if def.variant(data_idx).fields.len() == 1 {
            return def.variant(data_idx).single_field().ty(tcx, args);
        }
    }

    ty
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// FIXME: Move this check out of typeck, since it'll easily cycle when revealing opaques,
    /// and we shouldn't need to check anything here if the typeck results are tainted.
    pub(crate) fn check_transmute(&self, from: Ty<'tcx>, to: Ty<'tcx>, hir_id: HirId) {
        let tcx = self.tcx;
        let dl = &tcx.data_layout;
        let span = tcx.hir_span(hir_id);
        let normalize = |ty| {
            let ty = self.resolve_vars_if_possible(ty);
            if let Ok(ty) =
                self.tcx.try_normalize_erasing_regions(self.typing_env(self.param_env), ty)
            {
                ty
            } else {
                Ty::new_error_with_message(
                    tcx,
                    span,
                    "tried to normalize non-wf type in check_transmute",
                )
            }
        };
        let from = normalize(from);
        let to = normalize(to);
        trace!(?from, ?to);
        if from.has_non_region_infer() || to.has_non_region_infer() {
            // Note: this path is currently not reached in any test, so any
            // example that triggers this would be worth minimizing and
            // converting into a test.
            self.dcx().span_bug(span, "argument to transmute has inference variables");
        }
        // Transmutes that are only changing lifetimes are always ok.
        if from == to {
            return;
        }

        let skel = |ty| SizeSkeleton::compute(ty, tcx, self.typing_env(self.param_env));
        let sk_from = skel(from);
        let sk_to = skel(to);
        trace!(?sk_from, ?sk_to);

        // Check for same size using the skeletons.
        if let (Ok(sk_from), Ok(sk_to)) = (sk_from, sk_to) {
            if sk_from.same_size(sk_to) {
                return;
            }

            // Special-case transmuting from `typeof(function)` and
            // `Option<typeof(function)>` to present a clearer error.
            let from = unpack_option_like(tcx, from);
            if let (&ty::FnDef(..), SizeSkeleton::Known(size_to, _)) = (from.kind(), sk_to)
                && size_to == Pointer(dl.instruction_address_space).size(&tcx)
            {
                struct_span_code_err!(self.dcx(), span, E0591, "can't transmute zero-sized type")
                    .with_note(format!("source type: {from}"))
                    .with_note(format!("target type: {to}"))
                    .with_help("cast with `as` to a pointer instead")
                    .emit();
                return;
            }
        }

        // Try to display a sensible error with as much information as possible.
        let skeleton_string = |ty: Ty<'tcx>, sk: Result<_, &_>| match sk {
            Ok(SizeSkeleton::Pointer { tail, .. }) => format!("pointer to `{tail}`"),
            Ok(SizeSkeleton::Known(size, _)) => {
                if let Some(v) = u128::from(size.bytes()).checked_mul(8) {
                    format!("{v} bits")
                } else {
                    // `u128` should definitely be able to hold the size of different architectures
                    // larger sizes should be reported as error `are too big for the target architecture`
                    // otherwise we have a bug somewhere
                    bug!("{:?} overflow for u128", size)
                }
            }
            Ok(SizeSkeleton::Generic(size)) => {
                if let Some(size) =
                    self.try_structurally_resolve_const(span, size).try_to_target_usize(tcx)
                {
                    format!("{size} bytes")
                } else {
                    format!("generic size {size}")
                }
            }
            Err(LayoutError::TooGeneric(bad)) => {
                if *bad == ty {
                    "this type does not have a fixed size".to_owned()
                } else {
                    format!("size can vary because of {bad}")
                }
            }
            Err(err) => err.to_string(),
        };

        let mut err = struct_span_code_err!(
            self.dcx(),
            span,
            E0512,
            "cannot transmute between types of different sizes, \
                                        or dependently-sized types"
        );
        if from == to {
            err.note(format!("`{from}` does not have a fixed size"));
            err.emit();
        } else {
            err.note(format!("source type: `{}` ({})", from, skeleton_string(from, sk_from)))
                .note(format!("target type: `{}` ({})", to, skeleton_string(to, sk_to)));
            if let Err(LayoutError::ReferencesError(_)) = sk_from {
                err.delay_as_bug();
            } else if let Err(LayoutError::ReferencesError(_)) = sk_to {
                err.delay_as_bug();
            } else {
                err.emit();
            }
        }
    }
}
