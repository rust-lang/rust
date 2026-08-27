use rustc_abi::{FIRST_VARIANT, FieldIdx};
use rustc_attr_ir::lang_items::LangItem;
use rustc_hir as hir;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;

use crate::thir::cx::ThirBuildCx;

impl<'tcx> ThirBuildCx<'tcx> {
    fn push_place_expr(
        &mut self,
        temp_scope_id: hir::ItemLocalId,
        ty: Ty<'tcx>,
        span: Span,
        kind: ExprKind<'tcx>,
    ) -> ExprId {
        self.thir.exprs.push(Expr { temp_scope_id, ty, span, kind })
    }

    pub(super) fn mirror_raw_handle(
        &mut self,
        expr: &'tcx hir::Expr<'tcx>,
        pe: &'tcx hir::PlaceExpr<'tcx>,
    ) -> ExprKind<'tcx> {
        let tcx = self.tcx;
        let data = self
            .typeck_results
            .raw_handle_data()
            .get(expr.hir_id)
            .expect("no raw_handle_data for raw_handle");
        let temp_scope_id = expr.hir_id.local_id;
        let root_ty = data.root_ty;
        let root_span = pe.path_span;
        let res = self.typeck_results.qpath_res(&pe.path, expr.hir_id);
        let root_kind = self.convert_path(expr.hir_id, temp_scope_id, root_span, res);
        let arg = self.push_place_expr(temp_scope_id, root_ty, root_span, root_kind);

        let target_ptr_ty = Ty::new_mut_ptr(tcx, root_ty);
        let raw_expr = if data.mutability.is_mut() {
            self.push_place_expr(
                temp_scope_id,
                target_ptr_ty,
                root_span,
                ExprKind::RawBorrow { mutability: hir::Mutability::Mut, arg },
            )
        } else {
            let const_ptr_ty = Ty::new_imm_ptr(tcx, root_ty);
            let const_raw = self.push_place_expr(
                temp_scope_id,
                const_ptr_ty,
                root_span,
                ExprKind::RawBorrow { mutability: hir::Mutability::Not, arg },
            );
            // FIXME(#145383): Casting `*const T` to `*mut T` (`CastKind::PtrToPtr`) drops
            // pointee region constraints whenever `T` contains a region
            // (such as `let r = &mut x`).
            self.push_place_expr(
                temp_scope_id,
                target_ptr_ty,
                root_span,
                ExprKind::Cast { source: const_raw },
            )
        };

        let local_handle_did = tcx.require_lang_item(LangItem::LocalHandle, root_span);
        let local_handle_adt = tcx.adt_def(local_handle_did);
        let local_handle_args = tcx.mk_args(&[root_ty.into()]);
        let mut handle_ty = Ty::new_adt(tcx, local_handle_adt, local_handle_args);
        let mut cur_kind = ExprKind::Adt(Box::new(AdtExpr {
            adt_def: local_handle_adt,
            variant_index: FIRST_VARIANT,
            args: local_handle_args,
            fields: Box::new([FieldExpr { name: FieldIdx::from(0u32), expr: raw_expr }]),
            user_ty: None,
            base: AdtExprBase::None,
        }));

        let mut fields = data.fields.iter();
        for proj in pe.projections {
            (cur_kind, handle_ty) = match *proj {
                hir::PlaceExprKind::Deref(span) => self.lower_raw_handle_step(
                    temp_scope_id,
                    cur_kind,
                    handle_ty,
                    LangItem::DerefPlacePointeeHandle,
                    LangItem::DerefPlaceDerefPlace,
                    None,
                    span,
                ),
                hir::PlaceExprKind::Field(_, span) => {
                    let field = fields.next().expect("missing RawHandleField entry");
                    self.lower_raw_handle_field(temp_scope_id, span, field, cur_kind, handle_ty)
                }
            };
        }
        debug_assert!(fields.next().is_none());

        cur_kind
    }

    fn lower_raw_handle_step(
        &mut self,
        temp_scope_id: hir::ItemLocalId,
        cur_kind: ExprKind<'tcx>,
        handle_ty: Ty<'tcx>,
        assoc_item: LangItem,
        fn_item: LangItem,
        subplace_ty: Option<Ty<'tcx>>,
        span: Span,
    ) -> (ExprKind<'tcx>, Ty<'tcx>) {
        let tcx = self.tcx;
        let generic_args = match subplace_ty {
            Some(ty) => tcx.mk_args(&[handle_ty.into(), ty.into()]),
            None => tcx.mk_args(&[handle_ty.into()]),
        };
        let assoc_did = tcx.require_lang_item(assoc_item, span);
        let next_handle_ty = tcx.normalize_erasing_regions(
            self.typing_env,
            ty::Unnormalized::new_wip(Ty::new_projection_from_args(
                tcx,
                ty::IsRigid::No,
                assoc_did,
                generic_args,
            )),
        );
        let fn_did = tcx.require_lang_item(fn_item, span);
        let fun_ty = tcx.type_of(fn_did).instantiate(tcx, generic_args).skip_norm_wip();
        let fun = self.push_place_expr(
            temp_scope_id,
            fun_ty,
            span,
            ExprKind::ZstLiteral { user_ty: None },
        );
        let cur_expr = self.push_place_expr(temp_scope_id, handle_ty, span, cur_kind);
        let args: Box<[_]> = match subplace_ty {
            Some(ty) => {
                let subplace_expr = self.push_place_expr(
                    temp_scope_id,
                    ty,
                    span,
                    ExprKind::ZstLiteral { user_ty: None },
                );
                Box::new([cur_expr, subplace_expr])
            }
            None => Box::new([cur_expr]),
        };
        let next_kind =
            ExprKind::Call { ty: fun_ty, fun, args, from_hir_call: false, fn_span: span };
        (next_kind, next_handle_ty)
    }

    fn lower_raw_handle_field(
        &mut self,
        temp_scope_id: hir::ItemLocalId,
        span: Span,
        field: &ty::RawHandleField<'tcx>,
        mut cur_kind: ExprKind<'tcx>,
        mut handle_ty: Ty<'tcx>,
    ) -> (ExprKind<'tcx>, Ty<'tcx>) {
        for _ in 0..field.autoderefs {
            (cur_kind, handle_ty) = self.lower_raw_handle_step(
                temp_scope_id,
                cur_kind,
                handle_ty,
                LangItem::DerefPlacePointeeHandle,
                LangItem::DerefPlaceDerefPlace,
                None,
                span,
            );
        }
        let subplace_ty = Ty::new_field_representing_type(
            self.tcx,
            field.adjusted_base_ty,
            FIRST_VARIANT,
            field.field_idx,
        );
        self.lower_raw_handle_step(
            temp_scope_id,
            cur_kind,
            handle_ty,
            LangItem::ProjectPlaceProjected,
            LangItem::ProjectPlaceProjectPlace,
            Some(subplace_ty),
            span,
        )
    }
}
