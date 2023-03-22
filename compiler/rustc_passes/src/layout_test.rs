use rustc_ast::Attribute;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::layout::{HasParamEnv, HasTyCtxt, LayoutError, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::{ParamEnv, Ty, TyCtxt};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_target::abi::{HasDataLayout, TargetDataLayout};

use crate::errors::{Abi, Align, HomogeneousAggregate, LayoutOf, Size, UnrecognizedField};

pub fn test_layout(tcx: TyCtxt<'_>) {
    if tcx.features().rustc_attrs {
        // if the `rustc_attrs` feature is not enabled, don't bother testing layout
        for id in tcx.hir().items() {
            if matches!(
                tcx.def_kind(id.owner_id),
                DefKind::TyAlias | DefKind::Enum | DefKind::Struct | DefKind::Union
            ) {
                for attr in tcx.get_attrs(id.owner_id, sym::rustc_layout) {
                    dump_layout_of(tcx, id.owner_id.def_id, attr);
                }
            }
        }
    }
}

fn dump_layout_of(tcx: TyCtxt<'_>, item_def_id: LocalDefId, attr: &Attribute) {
    let tcx = tcx;
    let param_env = tcx.param_env(item_def_id);
    let ty = tcx.type_of(item_def_id).subst_identity();
    match tcx.layout_of(param_env.and(ty)) {
        Ok(ty_layout) => {
            // Check out the `#[rustc_layout(..)]` attribute to tell what to dump.
            // The `..` are the names of fields to dump.
            let meta_items = attr.meta_item_list().unwrap_or_default();
            for meta_item in meta_items {
                match meta_item.name_or_empty() {
                    sym::abi => {
                        tcx.sess.emit_err(Abi {
                            span: tcx.def_span(item_def_id.to_def_id()),
                            abi: format!("{:?}", ty_layout.abi),
                        });
                    }

                    sym::align => {
                        tcx.sess.emit_err(Align {
                            span: tcx.def_span(item_def_id.to_def_id()),
                            align: format!("{:?}", ty_layout.align),
                        });
                    }

                    sym::size => {
                        tcx.sess.emit_err(Size {
                            span: tcx.def_span(item_def_id.to_def_id()),
                            size: format!("{:?}", ty_layout.size),
                        });
                    }

                    sym::homogeneous_aggregate => {
                        tcx.sess.emit_err(HomogeneousAggregate {
                            span: tcx.def_span(item_def_id.to_def_id()),
                            homogeneous_aggregate: format!(
                                "{:?}",
                                ty_layout.homogeneous_aggregate(&UnwrapLayoutCx { tcx, param_env })
                            ),
                        });
                    }

                    sym::debug => {
                        let normalized_ty = format!(
                            "{:?}",
                            tcx.normalize_erasing_regions(
                                param_env.with_reveal_all_normalized(tcx),
                                ty,
                            )
                        );
                        let ty_layout = format!("{:#?}", *ty_layout);
                        tcx.sess.emit_err(LayoutOf {
                            span: tcx.def_span(item_def_id.to_def_id()),
                            normalized_ty,
                            ty_layout,
                        });
                    }

                    name => {
                        tcx.sess.emit_err(UnrecognizedField { span: meta_item.span(), name });
                    }
                }
            }
        }

        Err(layout_error) => {
            tcx.sess.emit_fatal(Spanned {
                node: layout_error,
                span: tcx.def_span(item_def_id.to_def_id()),
            });
        }
    }
}

struct UnwrapLayoutCx<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
}

impl<'tcx> LayoutOfHelpers<'tcx> for UnwrapLayoutCx<'tcx> {
    type LayoutOfResult = TyAndLayout<'tcx>;

    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        span_bug!(
            span,
            "`#[rustc_layout(..)]` test resulted in `layout_of({}) = Err({})`",
            ty,
            err
        );
    }
}

impl<'tcx> HasTyCtxt<'tcx> for UnwrapLayoutCx<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> HasParamEnv<'tcx> for UnwrapLayoutCx<'tcx> {
    fn param_env(&self) -> ParamEnv<'tcx> {
        self.param_env
    }
}

impl<'tcx> HasDataLayout for UnwrapLayoutCx<'tcx> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.tcx.data_layout()
    }
}
