//! MIR lowering for places

use hir_def::FunctionId;
use intern::sym;
use rustc_type_ir::inherent::{Region as _, Ty as _};

use super::*;
use crate::{
    mir::{MutBorrowKind, Operand, OperandKind},
    next_solver::Region,
};

macro_rules! not_supported {
    ($it: expr) => {
        return Err(MirLowerError::NotSupported(format!($it)))
    };
}

impl<'db> MirLowerCtx<'_, 'db> {
    fn lower_expr_to_some_place_without_adjust(
        &mut self,
        expr_id: ExprId,
        prev_block: BasicBlockId<'db>,
    ) -> Result<'db, Option<(Place<'db>, BasicBlockId<'db>)>> {
        let ty = self.expr_ty_without_adjust(expr_id);
        let place = self.temp(ty, prev_block, expr_id.into())?;
        let Some(current) =
            self.lower_expr_to_place_without_adjust(expr_id, place.into(), prev_block)?
        else {
            return Ok(None);
        };
        Ok(Some((place.into(), current)))
    }

    fn lower_expr_to_some_place_with_adjust(
        &mut self,
        expr_id: ExprId,
        prev_block: BasicBlockId<'db>,
        adjustments: &[Adjustment<'db>],
    ) -> Result<'db, Option<(Place<'db>, BasicBlockId<'db>)>> {
        let ty = adjustments
            .last()
            .map(|it| it.target)
            .unwrap_or_else(|| self.expr_ty_without_adjust(expr_id));
        let place = self.temp(ty, prev_block, expr_id.into())?;
        let Some(current) =
            self.lower_expr_to_place_with_adjust(expr_id, place.into(), prev_block, adjustments)?
        else {
            return Ok(None);
        };
        Ok(Some((place.into(), current)))
    }

    pub(super) fn lower_expr_as_place_with_adjust(
        &mut self,
        current: BasicBlockId<'db>,
        expr_id: ExprId,
        upgrade_rvalue: bool,
        adjustments: &[Adjustment<'db>],
    ) -> Result<'db, Option<(Place<'db>, BasicBlockId<'db>)>> {
        let try_rvalue = |this: &mut MirLowerCtx<'_, 'db>| {
            if !upgrade_rvalue {
                return Err(MirLowerError::MutatingRvalue);
            }
            this.lower_expr_to_some_place_with_adjust(expr_id, current, adjustments)
        };
        if let Some((last, rest)) = adjustments.split_last() {
            match last.kind {
                Adjust::Deref(None) => {
                    let Some(mut it) = self.lower_expr_as_place_with_adjust(
                        current,
                        expr_id,
                        upgrade_rvalue,
                        rest,
                    )?
                    else {
                        return Ok(None);
                    };
                    it.0 = it.0.project(ProjectionElem::Deref, &mut self.result.projection_store);
                    Ok(Some(it))
                }
                Adjust::Deref(Some(od)) => {
                    let Some((r, current)) = self.lower_expr_as_place_with_adjust(
                        current,
                        expr_id,
                        upgrade_rvalue,
                        rest,
                    )?
                    else {
                        return Ok(None);
                    };
                    self.lower_overloaded_deref(
                        current,
                        r,
                        rest.last()
                            .map(|it| it.target)
                            .unwrap_or_else(|| self.expr_ty_without_adjust(expr_id)),
                        last.target,
                        expr_id.into(),
                        match od.0 {
                            Some(Mutability::Mut) => true,
                            Some(Mutability::Not) => false,
                            None => {
                                not_supported!("implicit overloaded deref with unknown mutability")
                            }
                        },
                    )
                }
                Adjust::NeverToAny | Adjust::Borrow(_) | Adjust::Pointer(_) => try_rvalue(self),
            }
        } else {
            self.lower_expr_as_place_without_adjust(current, expr_id, upgrade_rvalue)
        }
    }

    pub(super) fn lower_expr_as_place(
        &mut self,
        current: BasicBlockId<'db>,
        expr_id: ExprId,
        upgrade_rvalue: bool,
    ) -> Result<'db, Option<(Place<'db>, BasicBlockId<'db>)>> {
        match self.infer.expr_adjustments.get(&expr_id) {
            Some(a) => self.lower_expr_as_place_with_adjust(current, expr_id, upgrade_rvalue, a),
            None => self.lower_expr_as_place_without_adjust(current, expr_id, upgrade_rvalue),
        }
    }

    pub(super) fn lower_expr_as_place_without_adjust(
        &mut self,
        current: BasicBlockId<'db>,
        expr_id: ExprId,
        upgrade_rvalue: bool,
    ) -> Result<'db, Option<(Place<'db>, BasicBlockId<'db>)>> {
        let try_rvalue = |this: &mut MirLowerCtx<'_, 'db>| {
            if !upgrade_rvalue {
                return Err(MirLowerError::MutatingRvalue);
            }
            this.lower_expr_to_some_place_without_adjust(expr_id, current)
        };
        match &self.body[expr_id] {
            Expr::Path(p) => {
                let resolver_guard =
                    self.resolver.update_to_inner_scope(self.db, self.owner, expr_id);
                let hygiene = self.body.expr_path_hygiene(expr_id);
                let resolved = self.resolver.resolve_path_in_value_ns_fully(self.db, p, hygiene);
                self.resolver.reset_to_guard(resolver_guard);
                let Some(pr) = resolved else {
                    return try_rvalue(self);
                };
                match pr {
                    ValueNs::LocalBinding(pat_id) => {
                        Ok(Some((self.binding_local(pat_id)?.into(), current)))
                    }
                    ValueNs::StaticId(s) => {
                        let ty = self.expr_ty_without_adjust(expr_id);
                        let ref_ty = Ty::new_ref(
                            self.interner(),
                            Region::new_static(self.interner()),
                            ty,
                            Mutability::Not,
                        );
                        let temp: Place<'db> = self.temp(ref_ty, current, expr_id.into())?.into();
                        self.push_assignment(
                            current,
                            temp,
                            Operand { kind: OperandKind::Static(s), span: None }.into(),
                            expr_id.into(),
                        );
                        Ok(Some((
                            temp.project(ProjectionElem::Deref, &mut self.result.projection_store),
                            current,
                        )))
                    }
                    _ => try_rvalue(self),
                }
            }
            Expr::UnaryOp { expr, op: hir_def::hir::UnaryOp::Deref } => {
                let is_builtin = match self.expr_ty_without_adjust(*expr).kind() {
                    TyKind::Ref(..) | TyKind::RawPtr(..) => true,
                    TyKind::Adt(id, _) => id.is_box(),
                    _ => false,
                };
                if !is_builtin {
                    let Some((p, current)) = self.lower_expr_as_place(current, *expr, true)? else {
                        return Ok(None);
                    };
                    return self.lower_overloaded_deref(
                        current,
                        p,
                        self.expr_ty_without_adjust(*expr),
                        self.expr_ty_without_adjust(expr_id),
                        expr_id.into(),
                        'b: {
                            if let Some((f, _)) = self.infer.method_resolution(expr_id)
                                && let Some(deref_trait) = self.lang_items().DerefMut
                                && let Some(deref_fn) = deref_trait
                                    .trait_items(self.db)
                                    .method_by_name(&Name::new_symbol_root(sym::deref_mut))
                            {
                                break 'b deref_fn == f;
                            }
                            false
                        },
                    );
                }
                let Some((mut r, current)) = self.lower_expr_as_place(current, *expr, true)? else {
                    return Ok(None);
                };
                r = r.project(ProjectionElem::Deref, &mut self.result.projection_store);
                Ok(Some((r, current)))
            }
            Expr::UnaryOp { .. } => try_rvalue(self),
            Expr::Field { expr, .. } => {
                let Some((mut r, current)) = self.lower_expr_as_place(current, *expr, true)? else {
                    return Ok(None);
                };
                self.push_field_projection(&mut r, expr_id)?;
                Ok(Some((r, current)))
            }
            Expr::Index { base, index } => {
                let base_ty = self.expr_ty_after_adjustments(*base);
                let index_ty = self.expr_ty_after_adjustments(*index);
                if !matches!(index_ty.kind(), TyKind::Uint(rustc_ast_ir::UintTy::Usize))
                    || !matches!(
                        base_ty.strip_reference().kind(),
                        TyKind::Array(..) | TyKind::Slice(..)
                    )
                {
                    let Some(index_fn) = self.infer.method_resolution(expr_id) else {
                        return Err(MirLowerError::UnresolvedMethod(
                            "[overloaded index]".to_owned(),
                        ));
                    };
                    let Some((base_place, current)) =
                        self.lower_expr_as_place(current, *base, true)?
                    else {
                        return Ok(None);
                    };
                    let Some((index_operand, current)) =
                        self.lower_expr_to_some_operand(*index, current)?
                    else {
                        return Ok(None);
                    };
                    return self.lower_overloaded_index(
                        current,
                        base_place,
                        base_ty,
                        self.expr_ty_without_adjust(expr_id),
                        index_operand,
                        expr_id.into(),
                        index_fn,
                    );
                }
                let adjusts = self
                    .infer
                    .expr_adjustments
                    .get(base)
                    .and_then(|it| it.split_last())
                    .map(|it| it.1)
                    .unwrap_or(&[]);
                let Some((mut p_base, current)) =
                    self.lower_expr_as_place_with_adjust(current, *base, true, adjusts)?
                else {
                    return Ok(None);
                };
                let l_index =
                    self.temp(self.expr_ty_after_adjustments(*index), current, expr_id.into())?;
                let Some(current) = self.lower_expr_to_place(*index, l_index.into(), current)?
                else {
                    return Ok(None);
                };
                p_base = p_base
                    .project(ProjectionElem::Index(l_index), &mut self.result.projection_store);
                Ok(Some((p_base, current)))
            }
            _ => try_rvalue(self),
        }
    }

    fn lower_overloaded_index(
        &mut self,
        current: BasicBlockId<'db>,
        place: Place<'db>,
        base_ty: Ty<'db>,
        result_ty: Ty<'db>,
        index_operand: Operand<'db>,
        span: MirSpan,
        index_fn: (FunctionId, GenericArgs<'db>),
    ) -> Result<'db, Option<(Place<'db>, BasicBlockId<'db>)>> {
        let mutability = match base_ty.as_reference() {
            Some((_, _, mutability)) => mutability,
            None => Mutability::Not,
        };
        let result_ref =
            Ty::new_ref(self.interner(), Region::error(self.interner()), result_ty, mutability);
        let mut result: Place<'db> = self.temp(result_ref, current, span)?.into();
        let index_fn_op = Operand::const_zst(Ty::new_fn_def(
            self.interner(),
            CallableDefId::FunctionId(index_fn.0).into(),
            index_fn.1,
        ));
        let Some(current) = self.lower_call(
            index_fn_op,
            Box::new([Operand { kind: OperandKind::Copy(place), span: None }, index_operand]),
            result,
            current,
            false,
            span,
        )?
        else {
            return Ok(None);
        };
        result = result.project(ProjectionElem::Deref, &mut self.result.projection_store);
        Ok(Some((result, current)))
    }

    fn lower_overloaded_deref(
        &mut self,
        current: BasicBlockId<'db>,
        place: Place<'db>,
        source_ty: Ty<'db>,
        target_ty: Ty<'db>,
        span: MirSpan,
        mutability: bool,
    ) -> Result<'db, Option<(Place<'db>, BasicBlockId<'db>)>> {
        let lang_items = self.lang_items();
        let (mutability, trait_lang_item, trait_method_name, borrow_kind) = if !mutability {
            (
                Mutability::Not,
                lang_items.Deref,
                Name::new_symbol_root(sym::deref),
                BorrowKind::Shared,
            )
        } else {
            (
                Mutability::Mut,
                lang_items.DerefMut,
                Name::new_symbol_root(sym::deref_mut),
                BorrowKind::Mut { kind: MutBorrowKind::Default },
            )
        };
        let error_region = Region::error(self.interner());
        let ty_ref = Ty::new_ref(self.interner(), error_region, source_ty, mutability);
        let target_ty_ref = Ty::new_ref(self.interner(), error_region, target_ty, mutability);
        let ref_place: Place<'db> = self.temp(ty_ref, current, span)?.into();
        self.push_assignment(current, ref_place, Rvalue::Ref(borrow_kind, place), span);
        let deref_trait = trait_lang_item.ok_or(MirLowerError::LangItemNotFound)?;
        let deref_fn = deref_trait
            .trait_items(self.db)
            .method_by_name(&trait_method_name)
            .ok_or(MirLowerError::LangItemNotFound)?;
        let deref_fn_op = Operand::const_zst(Ty::new_fn_def(
            self.interner(),
            CallableDefId::FunctionId(deref_fn).into(),
            GenericArgs::new_from_iter(self.interner(), [source_ty.into()]),
        ));
        let mut result: Place<'db> = self.temp(target_ty_ref, current, span)?.into();
        let Some(current) = self.lower_call(
            deref_fn_op,
            Box::new([Operand { kind: OperandKind::Copy(ref_place), span: None }]),
            result,
            current,
            false,
            span,
        )?
        else {
            return Ok(None);
        };
        result = result.project(ProjectionElem::Deref, &mut self.result.projection_store);
        Ok(Some((result, current)))
    }
}
