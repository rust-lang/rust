use crate::errors;
use rustc_ast::ast;
use rustc_attr::InstructionSetAttr;
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::Applicability;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::symbol::Symbol;
use rustc_span::Span;

pub fn from_target_feature(
    tcx: TyCtxt<'_>,
    attr: &ast::Attribute,
    supported_target_features: &UnordMap<String, Option<Symbol>>,
    target_features: &mut Vec<Symbol>,
) {
    let Some(list) = attr.meta_item_list() else { return };
    let bad_item = |span| {
        let msg = "malformed `target_feature` attribute input";
        let code = "enable = \"..\"";
        tcx.dcx()
            .struct_span_err(span, msg)
            .with_span_suggestion(span, "must be of the form", code, Applicability::HasPlaceholders)
            .emit();
    };
    let rust_features = tcx.features();
    for item in list {
        // Only `enable = ...` is accepted in the meta-item list.
        if !item.has_name(sym::enable) {
            bad_item(item.span());
            continue;
        }

        // Must be of the form `enable = "..."` (a string).
        let Some(value) = item.value_str() else {
            bad_item(item.span());
            continue;
        };

        // We allow comma separation to enable multiple features.
        target_features.extend(value.as_str().split(',').filter_map(|feature| {
            let Some(feature_gate) = supported_target_features.get(feature) else {
                let msg = format!("the feature named `{feature}` is not valid for this target");
                let mut err = tcx.dcx().struct_span_err(item.span(), msg);
                err.span_label(item.span(), format!("`{feature}` is not valid for this target"));
                if let Some(stripped) = feature.strip_prefix('+') {
                    let valid = supported_target_features.contains_key(stripped);
                    if valid {
                        err.help("consider removing the leading `+` in the feature name");
                    }
                }
                err.emit();
                return None;
            };

            // Only allow features whose feature gates have been enabled.
            let allowed = match feature_gate.as_ref().copied() {
                Some(sym::arm_target_feature) => rust_features.arm_target_feature,
                Some(sym::hexagon_target_feature) => rust_features.hexagon_target_feature,
                Some(sym::powerpc_target_feature) => rust_features.powerpc_target_feature,
                Some(sym::mips_target_feature) => rust_features.mips_target_feature,
                Some(sym::riscv_target_feature) => rust_features.riscv_target_feature,
                Some(sym::avx512_target_feature) => rust_features.avx512_target_feature,
                Some(sym::sse4a_target_feature) => rust_features.sse4a_target_feature,
                Some(sym::tbm_target_feature) => rust_features.tbm_target_feature,
                Some(sym::wasm_target_feature) => rust_features.wasm_target_feature,
                Some(sym::rtm_target_feature) => rust_features.rtm_target_feature,
                Some(sym::ermsb_target_feature) => rust_features.ermsb_target_feature,
                Some(sym::bpf_target_feature) => rust_features.bpf_target_feature,
                Some(sym::aarch64_ver_target_feature) => rust_features.aarch64_ver_target_feature,
                Some(sym::csky_target_feature) => rust_features.csky_target_feature,
                Some(sym::loongarch_target_feature) => rust_features.loongarch_target_feature,
                Some(sym::lahfsahf_target_feature) => rust_features.lahfsahf_target_feature,
                Some(sym::prfchw_target_feature) => rust_features.prfchw_target_feature,
                Some(name) => bug!("unknown target feature gate {}", name),
                None => true,
            };
            if !allowed {
                feature_err(
                    &tcx.sess,
                    feature_gate.unwrap(),
                    item.span(),
                    format!("the target feature `{feature}` is currently unstable"),
                )
                .emit();
            }
            Some(Symbol::intern(feature))
        }));
    }
}

/// Computes the set of target features used in a function for the purposes of
/// inline assembly.
fn asm_target_features(tcx: TyCtxt<'_>, did: DefId) -> &FxIndexSet<Symbol> {
    let mut target_features = tcx.sess.unstable_target_features.clone();
    if tcx.def_kind(did).has_codegen_attrs() {
        let attrs = tcx.codegen_fn_attrs(did);
        target_features.extend(&attrs.target_features);
        match attrs.instruction_set {
            None => {}
            Some(InstructionSetAttr::ArmA32) => {
                target_features.remove(&sym::thumb_mode);
            }
            Some(InstructionSetAttr::ArmT32) => {
                target_features.insert(sym::thumb_mode);
            }
        }
    }

    tcx.arena.alloc(target_features)
}

/// Checks the function annotated with `#[target_feature]` is not a safe
/// trait method implementation, reporting an error if it is.
pub fn check_target_feature_trait_unsafe(tcx: TyCtxt<'_>, id: LocalDefId, attr_span: Span) {
    if let DefKind::AssocFn = tcx.def_kind(id) {
        let parent_id = tcx.local_parent(id);
        if let DefKind::Trait | DefKind::Impl { of_trait: true } = tcx.def_kind(parent_id) {
            tcx.dcx().emit_err(errors::TargetFeatureSafeTrait {
                span: attr_span,
                def: tcx.def_span(id),
            });
        }
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        supported_target_features: |tcx, cnum| {
            assert_eq!(cnum, LOCAL_CRATE);
            if tcx.sess.opts.actually_rustdoc {
                // rustdoc needs to be able to document functions that use all the features, so
                // whitelist them all
                rustc_target::target_features::all_known_features()
                    .map(|(a, b)| (a.to_string(), b.as_feature_name()))
                    .collect()
            } else {
                tcx.sess
                    .target
                    .supported_target_features()
                    .iter()
                    .map(|&(a, b)| (a.to_string(), b.as_feature_name()))
                    .collect()
            }
        },
        asm_target_features,
        ..*providers
    }
}
