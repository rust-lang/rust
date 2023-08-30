use rustc_ast::Attribute;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::layout::{FnAbiError, LayoutError};
use rustc_middle::ty::{self, GenericArgs, Instance, Ty, TyCtxt};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::sym;
use rustc_target::abi::call::FnAbi;

use crate::errors::{AbiOf, UnrecognizedField};

pub fn test_abi(tcx: TyCtxt<'_>) {
    if !tcx.features().rustc_attrs {
        // if the `rustc_attrs` feature is not enabled, don't bother testing ABI
        return;
    }
    for id in tcx.hir().items() {
        match tcx.def_kind(id.owner_id) {
            DefKind::Fn => {
                for attr in tcx.get_attrs(id.owner_id, sym::rustc_abi) {
                    dump_abi_of_fn_item(tcx, id.owner_id.def_id.into(), attr);
                }
            }
            DefKind::TyAlias { .. } => {
                for attr in tcx.get_attrs(id.owner_id, sym::rustc_abi) {
                    dump_abi_of_fn_type(tcx, id.owner_id.def_id.into(), attr);
                }
            }
            DefKind::Impl { .. } => {
                // To find associated functions we need to go into the child items here.
                for &id in tcx.associated_item_def_ids(id.owner_id) {
                    if matches!(tcx.def_kind(id), DefKind::AssocFn) {
                        for attr in tcx.get_attrs(id, sym::rustc_abi) {
                            dump_abi_of_fn_item(tcx, id, attr);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn unwrap_fn_abi<'tcx>(
    abi: Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>>,
    tcx: TyCtxt<'tcx>,
    item_def_id: DefId,
) -> &'tcx FnAbi<'tcx, Ty<'tcx>> {
    match abi {
        Ok(abi) => abi,
        Err(FnAbiError::Layout(layout_error)) => {
            tcx.sess.emit_fatal(Spanned {
                node: layout_error.into_diagnostic(),
                span: tcx.def_span(item_def_id),
            });
        }
        Err(FnAbiError::AdjustForForeignAbi(e)) => {
            // Sadly there seems to be no `into_diagnostic` for this case... and I am not sure if
            // this can even be reached. Anyway this is a perma-unstable debug attribute, an ICE
            // isn't the worst thing. Also this matches what codegen does.
            span_bug!(
                tcx.def_span(item_def_id),
                "error computing fn_abi_of_instance, cannot adjust for foreign ABI: {e:?}",
            )
        }
    }
}

fn dump_abi_of_fn_item(tcx: TyCtxt<'_>, item_def_id: DefId, attr: &Attribute) {
    let param_env = tcx.param_env(item_def_id);
    let args = GenericArgs::identity_for_item(tcx, item_def_id);
    let instance = match Instance::resolve(tcx, param_env, item_def_id, args) {
        Ok(Some(instance)) => instance,
        Ok(None) => {
            // Not sure what to do here, but `LayoutError::Unknown` seems reasonable?
            let ty = tcx.type_of(item_def_id).instantiate_identity();
            tcx.sess.emit_fatal(Spanned {
                node: LayoutError::Unknown(ty).into_diagnostic(),

                span: tcx.def_span(item_def_id),
            });
        }
        Err(_guaranteed) => return,
    };
    let abi = unwrap_fn_abi(
        tcx.fn_abi_of_instance(param_env.and((instance, /* extra_args */ ty::List::empty()))),
        tcx,
        item_def_id,
    );

    // Check out the `#[rustc_abi(..)]` attribute to tell what to dump.
    // The `..` are the names of fields to dump.
    let meta_items = attr.meta_item_list().unwrap_or_default();
    for meta_item in meta_items {
        match meta_item.name_or_empty() {
            sym::debug => {
                let fn_name = tcx.item_name(item_def_id);
                tcx.sess.emit_err(AbiOf {
                    span: tcx.def_span(item_def_id),
                    fn_name,
                    fn_abi: format!("{:#?}", abi),
                });
            }

            name => {
                tcx.sess.emit_err(UnrecognizedField { span: meta_item.span(), name });
            }
        }
    }
}

fn dump_abi_of_fn_type(tcx: TyCtxt<'_>, item_def_id: DefId, attr: &Attribute) {
    let param_env = tcx.param_env(item_def_id);
    let ty = tcx.type_of(item_def_id).instantiate_identity();
    let meta_items = attr.meta_item_list().unwrap_or_default();
    for meta_item in meta_items {
        match meta_item.name_or_empty() {
            sym::debug => {
                let ty::FnPtr(sig) = ty.kind() else {
                    span_bug!(
                        meta_item.span(),
                        "`#[rustc_abi(debug)]` on a type alias requires function pointer type"
                    );
                };
                let abi = unwrap_fn_abi(
                    tcx.fn_abi_of_fn_ptr(param_env.and((*sig, /* extra_args */ ty::List::empty()))),
                    tcx,
                    item_def_id,
                );

                let fn_name = tcx.item_name(item_def_id);
                tcx.sess.emit_err(AbiOf {
                    span: tcx.def_span(item_def_id),
                    fn_name,
                    fn_abi: format!("{:#?}", abi),
                });
            }
            name => {
                tcx.sess.emit_err(UnrecognizedField { span: meta_item.span(), name });
            }
        }
    }
}
