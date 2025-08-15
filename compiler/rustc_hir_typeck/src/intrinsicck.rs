use hir::HirId;
use rustc_abi::Primitive::Pointer;
use rustc_abi::VariantIdx;
use rustc_errors::codes::*;
use rustc_errors::struct_span_code_err;
use rustc_hir as hir;
use rustc_index::Idx;
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutError, SizeSkeleton};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::LocalDefId;
use tracing::trace;

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

/// Try to display a sensible error with as much information as possible.
fn skeleton_string<'tcx>(
    ty: Ty<'tcx>,
    sk: Result<SizeSkeleton<'tcx>, &'tcx LayoutError<'tcx>>,
) -> String {
    match sk {
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
            format!("generic size {size}")
        }
        Err(LayoutError::TooGeneric(bad)) => {
            if *bad == ty {
                "this type does not have a fixed size".to_owned()
            } else {
                format!("size can vary because of {bad}")
            }
        }
        Err(err) => err.to_string(),
    }
}

fn check_transmute<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    from: Ty<'tcx>,
    to: Ty<'tcx>,
    hir_id: HirId,
) {
    let span = || tcx.hir_span(hir_id);
    let normalize = |ty| {
        if let Ok(ty) = tcx.try_normalize_erasing_regions(typing_env, ty) {
            ty
        } else {
            Ty::new_error_with_message(
                tcx,
                span(),
                format!("tried to normalize non-wf type {ty:#?} in check_transmute"),
            )
        }
    };

    let from = normalize(from);
    let to = normalize(to);
    trace!(?from, ?to);

    // Transmutes that are only changing lifetimes are always ok.
    if from == to {
        return;
    }

    let sk_from = SizeSkeleton::compute(from, tcx, typing_env);
    let sk_to = SizeSkeleton::compute(to, tcx, typing_env);
    trace!(?sk_from, ?sk_to);

    // Check for same size using the skeletons.
    if let Ok(sk_from) = sk_from
        && let Ok(sk_to) = sk_to
    {
        if sk_from.same_size(sk_to) {
            return;
        }

        // Special-case transmuting from `typeof(function)` and
        // `Option<typeof(function)>` to present a clearer error.
        let from = unpack_option_like(tcx, from);
        if let ty::FnDef(..) = from.kind()
            && let SizeSkeleton::Known(size_to, _) = sk_to
            && size_to == Pointer(tcx.data_layout.instruction_address_space).size(&tcx)
        {
            struct_span_code_err!(tcx.sess.dcx(), span(), E0591, "can't transmute zero-sized type")
                .with_note(format!("source type: {from}"))
                .with_note(format!("target type: {to}"))
                .with_help("cast with `as` to a pointer instead")
                .emit();
            return;
        }
    }

    let mut err = struct_span_code_err!(
        tcx.sess.dcx(),
        span(),
        E0512,
        "cannot transmute between types of different sizes, or dependently-sized types"
    );
    if from == to {
        err.note(format!("`{from}` does not have a fixed size"));
        err.emit();
    } else {
        err.note(format!("source type: `{}` ({})", from, skeleton_string(from, sk_from)));
        err.note(format!("target type: `{}` ({})", to, skeleton_string(to, sk_to)));
        err.emit();
    }
}

pub(crate) fn check_transmutes(tcx: TyCtxt<'_>, owner: LocalDefId) {
    assert!(!tcx.is_typeck_child(owner.to_def_id()));
    let typeck_results = tcx.typeck(owner);
    let None = typeck_results.tainted_by_errors else { return };

    let typing_env = ty::TypingEnv::post_analysis(tcx, owner);
    for &(from, to, hir_id) in &typeck_results.transmutes_to_check {
        check_transmute(tcx, typing_env, from, to, hir_id);
    }
}
