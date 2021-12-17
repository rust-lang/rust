use rustc_ast::Attribute;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::ItemKind;
use rustc_middle::ty::layout::{HasParamEnv, HasTyCtxt, LayoutError, LayoutOfHelpers, TyAndLayout};
use rustc_middle::ty::{ParamEnv, Ty, TyCtxt};
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_target::abi::{HasDataLayout, TargetDataLayout};

pub fn test_layout(tcx: TyCtxt<'_>) {
    if tcx.features().rustc_attrs {
        // if the `rustc_attrs` feature is not enabled, don't bother testing layout
        tcx.hir().visit_all_item_likes(&mut LayoutTest { tcx });
    }
}

struct LayoutTest<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> ItemLikeVisitor<'tcx> for LayoutTest<'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        match item.kind {
            ItemKind::TyAlias(..)
            | ItemKind::Enum(..)
            | ItemKind::Struct(..)
            | ItemKind::Union(..) => {
                for attr in self.tcx.get_attrs(item.def_id.to_def_id()).iter() {
                    if attr.has_name(sym::rustc_layout) {
                        self.dump_layout_of(item.def_id, item, attr);
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_trait_item(&mut self, _: &'tcx hir::TraitItem<'tcx>) {}
    fn visit_impl_item(&mut self, _: &'tcx hir::ImplItem<'tcx>) {}
    fn visit_foreign_item(&mut self, _: &'tcx hir::ForeignItem<'tcx>) {}
}

impl<'tcx> LayoutTest<'tcx> {
    fn dump_layout_of(&self, item_def_id: LocalDefId, item: &hir::Item<'tcx>, attr: &Attribute) {
        let tcx = self.tcx;
        let param_env = self.tcx.param_env(item_def_id);
        let ty = self.tcx.type_of(item_def_id);
        match self.tcx.layout_of(param_env.and(ty)) {
            Ok(ty_layout) => {
                // Check out the `#[rustc_layout(..)]` attribute to tell what to dump.
                // The `..` are the names of fields to dump.
                let meta_items = attr.meta_item_list().unwrap_or_default();
                for meta_item in meta_items {
                    match meta_item.name_or_empty() {
                        sym::abi => {
                            self.tcx.sess.span_err(item.span, &format!("abi: {:?}", ty_layout.abi));
                        }

                        sym::align => {
                            self.tcx
                                .sess
                                .span_err(item.span, &format!("align: {:?}", ty_layout.align));
                        }

                        sym::size => {
                            self.tcx
                                .sess
                                .span_err(item.span, &format!("size: {:?}", ty_layout.size));
                        }

                        sym::homogeneous_aggregate => {
                            self.tcx.sess.span_err(
                                item.span,
                                &format!(
                                    "homogeneous_aggregate: {:?}",
                                    ty_layout
                                        .homogeneous_aggregate(&UnwrapLayoutCx { tcx, param_env }),
                                ),
                            );
                        }

                        sym::debug => {
                            let normalized_ty = self.tcx.normalize_erasing_regions(
                                param_env.with_reveal_all_normalized(self.tcx),
                                ty,
                            );
                            self.tcx.sess.span_err(
                                item.span,
                                &format!("layout_of({:?}) = {:#?}", normalized_ty, *ty_layout),
                            );
                        }

                        name => {
                            self.tcx.sess.span_err(
                                meta_item.span(),
                                &format!("unrecognized field name `{}`", name),
                            );
                        }
                    }
                }
            }

            Err(layout_error) => {
                self.tcx.sess.span_err(item.span, &format!("layout error: {:?}", layout_error));
            }
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
