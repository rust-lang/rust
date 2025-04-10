use rustc_hir::Attribute;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::{FnAbiError, LayoutError};
use rustc_middle::ty::{self, GenericArgs, Instance, Ty, TyCtxt};
use rustc_span::source_map::Spanned;
use rustc_span::sym;
use rustc_target::callconv::FnAbi;

use super::layout_test::ensure_wf;
use crate::errors::{AbiInvalidAttribute, AbiNe, AbiOf, UnrecognizedArgument};

pub fn test_abi(tcx: TyCtxt<'_>) {
    if !tcx.features().rustc_attrs() {
        // if the `rustc_attrs` feature is not enabled, don't bother testing ABI
        return;
    }
    for id in tcx.hir_crate_items(()).definitions() {
        for attr in tcx.get_attrs(id, sym::rustc_abi) {
            match tcx.def_kind(id) {
                DefKind::Fn | DefKind::AssocFn => {
                    dump_abi_of_fn_item(tcx, id, attr);
                }
                DefKind::TyAlias => {
                    dump_abi_of_fn_type(tcx, id, attr);
                }
                _ => {
                    tcx.dcx().emit_err(AbiInvalidAttribute { span: tcx.def_span(id) });
                }
            }
        }
    }
}

fn unwrap_fn_abi<'tcx>(
    abi: Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>>,
    tcx: TyCtxt<'tcx>,
    item_def_id: LocalDefId,
) -> &'tcx FnAbi<'tcx, Ty<'tcx>> {
    match abi {
        Ok(abi) => abi,
        Err(FnAbiError::Layout(layout_error)) => {
            tcx.dcx().emit_fatal(Spanned {
                node: layout_error.into_diagnostic(),
                span: tcx.def_span(item_def_id),
            });
        }
    }
}

fn dump_abi_of_fn_item(tcx: TyCtxt<'_>, item_def_id: LocalDefId, attr: &Attribute) {
    let typing_env = ty::TypingEnv::post_analysis(tcx, item_def_id);
    let args = GenericArgs::identity_for_item(tcx, item_def_id);
    let instance = match Instance::try_resolve(tcx, typing_env, item_def_id.into(), args) {
        Ok(Some(instance)) => instance,
        Ok(None) => {
            // Not sure what to do here, but `LayoutError::Unknown` seems reasonable?
            let ty = tcx.type_of(item_def_id).instantiate_identity();
            tcx.dcx().emit_fatal(Spanned {
                node: LayoutError::Unknown(ty).into_diagnostic(),

                span: tcx.def_span(item_def_id),
            });
        }
        Err(_guaranteed) => return,
    };
    let abi = unwrap_fn_abi(
        tcx.fn_abi_of_instance(
            typing_env.as_query_input((instance, /* extra_args */ ty::List::empty())),
        ),
        tcx,
        item_def_id,
    );

    // Check out the `#[rustc_abi(..)]` attribute to tell what to dump.
    // The `..` are the names of fields to dump.
    let meta_items = attr.meta_item_list().unwrap_or_default();
    for meta_item in meta_items {
        match meta_item.name() {
            Some(sym::debug) => {
                let fn_name = tcx.item_name(item_def_id.into());
                tcx.dcx().emit_err(AbiOf {
                    span: tcx.def_span(item_def_id),
                    fn_name,
                    // FIXME: using the `Debug` impl here isn't ideal.
                    fn_abi: format!("{:#?}", abi),
                });
            }

            _ => {
                tcx.dcx().emit_err(UnrecognizedArgument { span: meta_item.span() });
            }
        }
    }
}

fn test_abi_eq<'tcx>(abi1: &'tcx FnAbi<'tcx, Ty<'tcx>>, abi2: &'tcx FnAbi<'tcx, Ty<'tcx>>) -> bool {
    if abi1.conv != abi2.conv
        || abi1.args.len() != abi2.args.len()
        || abi1.c_variadic != abi2.c_variadic
        || abi1.fixed_count != abi2.fixed_count
        || abi1.can_unwind != abi2.can_unwind
    {
        return false;
    }

    abi1.ret.eq_abi(&abi2.ret)
        && abi1.args.iter().zip(abi2.args.iter()).all(|(arg1, arg2)| arg1.eq_abi(arg2))
}

fn dump_abi_of_fn_type(tcx: TyCtxt<'_>, item_def_id: LocalDefId, attr: &Attribute) {
    let typing_env = ty::TypingEnv::post_analysis(tcx, item_def_id);
    let ty = tcx.type_of(item_def_id).instantiate_identity();
    let span = tcx.def_span(item_def_id);
    if !ensure_wf(tcx, typing_env, ty, item_def_id, span) {
        return;
    }
    let meta_items = attr.meta_item_list().unwrap_or_default();
    for meta_item in meta_items {
        match meta_item.name() {
            Some(sym::debug) => {
                let ty::FnPtr(sig_tys, hdr) = ty.kind() else {
                    span_bug!(
                        meta_item.span(),
                        "`#[rustc_abi(debug)]` on a type alias requires function pointer type"
                    );
                };
                let abi = unwrap_fn_abi(
                    tcx.fn_abi_of_fn_ptr(typing_env.as_query_input((
                        sig_tys.with(*hdr),
                        /* extra_args */ ty::List::empty(),
                    ))),
                    tcx,
                    item_def_id,
                );

                let fn_name = tcx.item_name(item_def_id.into());
                tcx.dcx().emit_err(AbiOf { span, fn_name, fn_abi: format!("{:#?}", abi) });
            }
            Some(sym::assert_eq) => {
                let ty::Tuple(fields) = ty.kind() else {
                    span_bug!(
                        meta_item.span(),
                        "`#[rustc_abi(assert_eq)]` on a type alias requires pair type"
                    );
                };
                let [field1, field2] = ***fields else {
                    span_bug!(
                        meta_item.span(),
                        "`#[rustc_abi(assert_eq)]` on a type alias requires pair type"
                    );
                };
                let ty::FnPtr(sig_tys1, hdr1) = field1.kind() else {
                    span_bug!(
                        meta_item.span(),
                        "`#[rustc_abi(assert_eq)]` on a type alias requires pair of function pointer types"
                    );
                };
                let abi1 = unwrap_fn_abi(
                    tcx.fn_abi_of_fn_ptr(typing_env.as_query_input((
                        sig_tys1.with(*hdr1),
                        /* extra_args */ ty::List::empty(),
                    ))),
                    tcx,
                    item_def_id,
                );
                let ty::FnPtr(sig_tys2, hdr2) = field2.kind() else {
                    span_bug!(
                        meta_item.span(),
                        "`#[rustc_abi(assert_eq)]` on a type alias requires pair of function pointer types"
                    );
                };
                let abi2 = unwrap_fn_abi(
                    tcx.fn_abi_of_fn_ptr(typing_env.as_query_input((
                        sig_tys2.with(*hdr2),
                        /* extra_args */ ty::List::empty(),
                    ))),
                    tcx,
                    item_def_id,
                );

                if !test_abi_eq(abi1, abi2) {
                    tcx.dcx().emit_err(AbiNe {
                        span,
                        left: format!("{:#?}", abi1),
                        right: format!("{:#?}", abi2),
                    });
                }
            }
            _ => {
                tcx.dcx().emit_err(UnrecognizedArgument { span: meta_item.span() });
            }
        }
    }
}
