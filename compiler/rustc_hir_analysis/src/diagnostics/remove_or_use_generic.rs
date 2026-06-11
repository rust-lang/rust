use std::ops::ControlFlow;

use rustc_errors::{Applicability, Diag};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor, walk_lifetime};
use rustc_hir::{GenericArg, HirId, LifetimeKind, Path, QPath, TyKind};
use rustc_middle::hir::nested_filter::All;
use rustc_middle::ty::{GenericParamDef, GenericParamDefKind, TyCtxt};

use crate::hir::def::Res;

/// Use a Visitor to find usages of the type or lifetime parameter
struct ParamUsageVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// The `DefId` of the generic parameter we are looking for.
    param_def_id: DefId,
    found: bool,
}

impl<'tcx> Visitor<'tcx> for ParamUsageVisitor<'tcx> {
    type NestedFilter = All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    type Result = ControlFlow<()>;

    fn visit_path(&mut self, path: &Path<'tcx>, _id: HirId) -> Self::Result {
        if let Some(res_def_id) = path.res.opt_def_id() {
            if res_def_id == self.param_def_id {
                self.found = true;
                return ControlFlow::Break(());
            }
        }
        intravisit::walk_path(self, path)
    }

    fn visit_lifetime(&mut self, lifetime: &'tcx rustc_hir::Lifetime) -> Self::Result {
        if let LifetimeKind::Param(id) = lifetime.kind {
            if let Some(local_def_id) = self.param_def_id.as_local() {
                if id == local_def_id {
                    self.found = true;
                    return ControlFlow::Break(());
                }
            }
        }
        walk_lifetime(self, lifetime)
    }
}

/// Adds a suggestion to a diagnostic to either remove an unused generic parameter, or use it.
///
/// # Examples
///
/// - `impl<T> Struct { ... }` where `T` is unused -> suggests removing `T` or using it.
/// - `impl<T> Struct { // T used in here }` where `T` is used in the body but not in the self type -> suggests adding `T` to the self type and struct definition.
/// - `impl<T> Struct { ... }` where the struct has a generic parameter with a default -> suggests adding `T` to the self type.
pub(crate) fn suggest_to_remove_or_use_generic(
    tcx: TyCtxt<'_>,
    diag: &mut Diag<'_>,
    impl_def_id: LocalDefId,
    param: &GenericParamDef,
    is_lifetime: bool,
) {
    let node = tcx.hir_node_by_def_id(impl_def_id);
    let hir_impl = node.expect_item().expect_impl();

    let Some((index, _)) = hir_impl
        .generics
        .params
        .iter()
        .enumerate()
        .find(|(_, par)| par.def_id.to_def_id() == param.def_id)
    else {
        return;
    };

    // Get the Struct/ADT definition ID from the self type
    let struct_def_id = if let TyKind::Path(QPath::Resolved(_, path)) = hir_impl.self_ty.kind
        && let Res::Def(DefKind::Struct | DefKind::Enum | DefKind::Union, def_id) = path.res
    {
        def_id
    } else {
        return;
    };

    // Count how many generic parameters are defined in the struct definition
    let generics = tcx.generics_of(struct_def_id);
    let total_params = generics
        .own_params
        .iter()
        .filter(|p| {
            if is_lifetime {
                matches!(p.kind, GenericParamDefKind::Lifetime)
            } else {
                matches!(p.kind, GenericParamDefKind::Type { .. })
            }
        })
        .count();

    // Count how many arguments are currently provided in the impl
    let mut provided_params = 0;
    let mut last_segment_args = None;

    if let TyKind::Path(QPath::Resolved(_, path)) = hir_impl.self_ty.kind
        && let Some(seg) = path.segments.last()
        && let Some(args) = seg.args
    {
        last_segment_args = Some(args);
        provided_params = args
            .args
            .iter()
            .filter(|arg| match arg {
                GenericArg::Lifetime(_) => is_lifetime,
                GenericArg::Type(_) => !is_lifetime,
                _ => false,
            })
            .count();
    }

    let mut visitor = ParamUsageVisitor { tcx, param_def_id: param.def_id, found: false };
    for item_ref in hir_impl.items {
        let _ = visitor.visit_impl_item_ref(item_ref);
        if visitor.found {
            break;
        }
    }
    let is_param_used = visitor.found;

    let mut suggestions = vec![];

    // Option A: Remove (Only if not used in body)
    if !is_param_used {
        suggestions.push((hir_impl.generics.span_for_param_removal(index), String::new()));
    }

    // Option B: Suggest adding only if there's an available parameter in the struct definition
    // or the parameter is already used somewhere, then we suggest adding to the impl struct and the struct definition
    if provided_params < total_params || is_param_used {
        if let Some(args) = last_segment_args {
            // Struct already has <...>, append to it
            suggestions.push((args.span().unwrap().shrink_to_hi(), format!(", {}", param.name)));
        } else if let TyKind::Path(QPath::Resolved(_, path)) = hir_impl.self_ty.kind {
            // Struct has no <...> yet, add it
            let seg = path.segments.last().unwrap();
            suggestions.push((seg.ident.span.shrink_to_hi(), format!("<{}>", param.name)));
        }
        if is_param_used {
            // If the parameter is used in the body, we also want to suggest adding it to the struct definition if it's not already there
            let struct_span = tcx.def_span(struct_def_id);
            let last_param_span = if let Some(local_def_id) = struct_def_id.as_local() {
                let hir_struct = tcx.hir_node_by_def_id(local_def_id).expect_item().expect_struct();
                hir_struct.1.params.last().map(|param| param.span)
            } else {
                let generics = tcx.generics_of(struct_def_id);
                generics.own_params.last().map(|param| tcx.def_span(param.def_id))
            };

            if let Some(last_param_span) = last_param_span {
                suggestions.push((last_param_span.shrink_to_hi(), format!(", {}", param.name)));
            } else {
                suggestions.push((struct_span.shrink_to_hi(), format!("<{}>", param.name)));
            }
        }
    }

    if suggestions.is_empty() {
        return;
    }

    let parameter_type = if is_lifetime { "lifetime" } else { "type" };
    if is_param_used {
        let msg = format!(
            "use the {} parameter `{}` in the `{}` type and use it in the type definition",
            parameter_type,
            param.name,
            tcx.def_path_str(struct_def_id)
        );
        diag.multipart_suggestion(
            msg,
            vec![
                (suggestions[0].0, suggestions[0].1.clone()),
                (suggestions[1].0, suggestions[1].1.clone()),
            ],
            Applicability::MaybeIncorrect,
        );
    } else {
        let msg = if suggestions.len() == 2 {
            format!("either remove the unused {} parameter `{}`", parameter_type, param.name)
        } else {
            format!("remove the unused {} parameter `{}`", parameter_type, param.name)
        };
        diag.span_suggestion(
            suggestions[0].0,
            msg,
            suggestions[0].1.clone(),
            Applicability::MaybeIncorrect,
        );
        if suggestions.len() == 2 {
            let msg = format!("or use it");
            diag.span_suggestion(
                suggestions[1].0,
                msg,
                suggestions[1].1.clone(),
                Applicability::MaybeIncorrect,
            );
        }
    };
}
