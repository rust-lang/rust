use rustc_hir::attrs::RustcDumpPtrauthDiscriminatorKind;
use rustc_hir::def::DefKind;
use rustc_hir::find_attr;
use rustc_middle::ptrauth::FnPtrTypeDiscriminatorInput;
use rustc_middle::ptrauth::discriminator::{
    compute_fn_ptr_type_discriminator, debug_encode_fn_ptr_type,
};
use rustc_middle::ty::TyCtxt;

pub fn test_ptrauth_discriminator(tcx: TyCtxt<'_>) {
    if !tcx.features().rustc_attrs() {
        return;
    }
    for def_id in tcx.hir_crate_items(()).definitions() {
        if tcx.def_kind(def_id) != DefKind::Fn {
            continue;
        }
        let Some(kinds) = find_attr!(tcx, def_id, RustcDumpPtrauthDiscriminator(kinds) => kinds)
        else {
            continue;
        };

        let sig = tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        let input = FnPtrTypeDiscriminatorInput::from_sig(sig);
        let span = tcx.def_span(def_id);

        for kind in kinds {
            let message = match kind {
                RustcDumpPtrauthDiscriminatorKind::Encoding => {
                    format!(
                        "ptrauth discriminator encoding: \"{}\"",
                        debug_encode_fn_ptr_type(tcx, &input)
                    )
                }
                RustcDumpPtrauthDiscriminatorKind::Hash => {
                    let res = compute_fn_ptr_type_discriminator(tcx, &input);
                    format!("ptrauth discriminator hash: {} (0x{:x})", res, res)
                }
            };
            tcx.dcx().span_err(span, message);
        }
    }
}
