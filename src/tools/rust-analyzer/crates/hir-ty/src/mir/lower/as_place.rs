//! MIR lowering for places

use super::*;
use hir_def::{lang_item::lang_attr, FunctionId};
use hir_expand::name;

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirLowerError::NotSupported(format!($x)))
    };
}

impl MirLowerCtx<'_> {
    fn lower_expr_to_some_place_without_adjust(
        &mut self,
        expr_id: ExprId,
        prev_block: BasicBlockId,
    ) -> Result<Option<(Place, BasicBlockId)>> {
        let ty = self.expr_ty_without_adjust(expr_id);
        let place = self.temp(ty, prev_block, expr_id.into())?;
        let Some(current) = self.lower_expr_to_place_without_adjust(expr_id, place.into(), prev_block)? else {
            return Ok(None);
        };
        Ok(Some((place.into(), current)))
    }

    fn lower_expr_to_some_place_with_adjust(
        &mut self,
        expr_id: ExprId,
        prev_block: BasicBlockId,
        adjustments: &[Adjustment],
    ) -> Result<Option<(Place, BasicBlockId)>> {
        let ty = adjustments
            .last()
            .map(|x| x.target.clone())
            .unwrap_or_else(|| self.expr_ty_without_adjust(expr_id));
        let place = self.temp(ty, prev_block, expr_id.into())?;
        let Some(current) = self.lower_expr_to_place_with_adjust(expr_id, place.into(), prev_block, adjustments)? else {
            return Ok(None);
        };
        Ok(Some((place.into(), current)))
    }

    pub(super) fn lower_expr_as_place_with_adjust(
        &mut self,
        current: BasicBlockId,
        expr_id: ExprId,
        upgrade_rvalue: bool,
        adjustments: &[Adjustment],
    ) -> Result<Option<(Place, BasicBlockId)>> {
        let try_rvalue = |this: &mut MirLowerCtx<'_>| {
            if !upgrade_rvalue {
                return Err(MirLowerError::MutatingRvalue);
            }
            this.lower_expr_to_some_place_with_adjust(expr_id, current, adjustments)
        };
        if let Some((last, rest)) = adjustments.split_last() {
            match last.kind {
                Adjust::Deref(None) => {
                    let Some(mut x) = self.lower_expr_as_place_with_adjust(
                        current,
                        expr_id,
                        upgrade_rvalue,
                        rest,
                    )? else {
                        return Ok(None);
                    };
                    x.0 = x.0.project(ProjectionElem::Deref);
                    Ok(Some(x))
                }
                Adjust::Deref(Some(od)) => {
                    let Some((r, current)) = self.lower_expr_as_place_with_adjust(
                        current,
                        expr_id,
                        upgrade_rvalue,
                        rest,
                    )? else {
                        return Ok(None);
                    };
                    self.lower_overloaded_deref(
                        current,
                        r,
                        rest.last()
                            .map(|x| x.target.clone())
                            .unwrap_or_else(|| self.expr_ty_without_adjust(expr_id)),
                        last.target.clone(),
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
        current: BasicBlockId,
        expr_id: ExprId,
        upgrade_rvalue: bool,
    ) -> Result<Option<(Place, BasicBlockId)>> {
        match self.infer.expr_adjustments.get(&expr_id) {
            Some(a) => self.lower_expr_as_place_with_adjust(current, expr_id, upgrade_rvalue, a),
            None => self.lower_expr_as_place_without_adjust(current, expr_id, upgrade_rvalue),
        }
    }

    pub(super) fn lower_expr_as_place_without_adjust(
        &mut self,
        current: BasicBlockId,
        expr_id: ExprId,
        upgrade_rvalue: bool,
    ) -> Result<Option<(Place, BasicBlockId)>> {
        let try_rvalue = |this: &mut MirLowerCtx<'_>| {
            if !upgrade_rvalue {
                return Err(MirLowerError::MutatingRvalue);
            }
            this.lower_expr_to_some_place_without_adjust(expr_id, current)
        };
        match &self.body.exprs[expr_id] {
            Expr::Path(p) => {
                let resolver = resolver_for_expr(self.db.upcast(), self.owner, expr_id);
                let Some(pr) = resolver.resolve_path_in_value_ns_fully(self.db.upcast(), p) else {
                    return try_rvalue(self);
                };
                match pr {
                    ValueNs::LocalBinding(pat_id) => {
                        Ok(Some((self.binding_local(pat_id)?.into(), current)))
                    }
                    ValueNs::StaticId(s) => {
                        let ty = self.expr_ty_without_adjust(expr_id);
                        let ref_ty =
                            TyKind::Ref(Mutability::Not, static_lifetime(), ty).intern(Interner);
                        let temp: Place = self.temp(ref_ty, current, expr_id.into())?.into();
                        self.push_assignment(
                            current,
                            temp.clone(),
                            Operand::Static(s).into(),
                            expr_id.into(),
                        );
                        Ok(Some((temp.project(ProjectionElem::Deref), current)))
                    }
                    _ => try_rvalue(self),
                }
            }
            Expr::UnaryOp { expr, op } => match op {
                hir_def::hir::UnaryOp::Deref => {
                    let is_builtin = match self.expr_ty_without_adjust(*expr).kind(Interner) {
                        TyKind::Ref(..) | TyKind::Raw(..) => true,
                        TyKind::Adt(id, _) => {
                            if let Some(lang_item) = lang_attr(self.db.upcast(), id.0) {
                                lang_item == LangItem::OwnedBox
                            } else {
                                false
                            }
                        }
                        _ => false,
                    };
                    if !is_builtin {
                        let Some((p, current)) = self.lower_expr_as_place(current, *expr, true)? else {
                            return Ok(None);
                        };
                        return self.lower_overloaded_deref(
                            current,
                            p,
                            self.expr_ty_after_adjustments(*expr),
                            self.expr_ty_without_adjust(expr_id),
                            expr_id.into(),
                            'b: {
                                if let Some((f, _)) = self.infer.method_resolution(expr_id) {
                                    if let Some(deref_trait) =
                                        self.resolve_lang_item(LangItem::DerefMut)?.as_trait()
                                    {
                                        if let Some(deref_fn) = self
                                            .db
                                            .trait_data(deref_trait)
                                            .method_by_name(&name![deref_mut])
                                        {
                                            break 'b deref_fn == f;
                                        }
                                    }
                                }
                                false
                            },
                        );
                    }
                    let Some((mut r, current)) = self.lower_expr_as_place(current, *expr, true)? else {
                        return Ok(None);
                    };
                    r = r.project(ProjectionElem::Deref);
                    Ok(Some((r, current)))
                }
                _ => try_rvalue(self),
            },
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
                if index_ty != TyBuilder::usize()
                    || !matches!(
                        base_ty.strip_reference().kind(Interner),
                        TyKind::Array(..) | TyKind::Slice(..)
                    )
                {
                    let Some(index_fn) = self.infer.method_resolution(expr_id) else {
                        return Err(MirLowerError::UnresolvedMethod("[overloaded index]".to_string()));
                    };
                    let Some((base_place, current)) = self.lower_expr_as_place(current, *base, true)? else {
                        return Ok(None);
                    };
                    let Some((index_operand, current)) = self.lower_expr_to_some_operand(*index, current)? else {
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
                    .and_then(|x| x.split_last())
                    .map(|x| x.1)
                    .unwrap_or(&[]);
                let Some((mut p_base, current)) =
                    self.lower_expr_as_place_with_adjust(current, *base, true, adjusts)?
                else {
                    return Ok(None);
                };
                let l_index =
                    self.temp(self.expr_ty_after_adjustments(*index), current, expr_id.into())?;
                let Some(current) = self.lower_expr_to_place(*index, l_index.into(), current)? else {
                    return Ok(None);
                };
                p_base = p_base.project(ProjectionElem::Index(l_index));
                Ok(Some((p_base, current)))
            }
            _ => try_rvalue(self),
        }
    }

    fn lower_overloaded_index(
        &mut self,
        current: BasicBlockId,
        place: Place,
        base_ty: Ty,
        result_ty: Ty,
        index_operand: Operand,
        span: MirSpan,
        index_fn: (FunctionId, Substitution),
    ) -> Result<Option<(Place, BasicBlockId)>> {
        let mutability = match base_ty.as_reference() {
            Some((_, _, mutability)) => mutability,
            None => Mutability::Not,
        };
        let result_ref = TyKind::Ref(mutability, static_lifetime(), result_ty).intern(Interner);
        let mut result: Place = self.temp(result_ref, current, span)?.into();
        let index_fn_op = Operand::const_zst(
            TyKind::FnDef(
                self.db.intern_callable_def(CallableDefId::FunctionId(index_fn.0)).into(),
                index_fn.1,
            )
            .intern(Interner),
        );
        let Some(current) = self.lower_call(index_fn_op, Box::new([Operand::Copy(place), index_operand]), result.clone(), current, false, span)? else {
            return Ok(None);
        };
        result = result.project(ProjectionElem::Deref);
        Ok(Some((result, current)))
    }

    fn lower_overloaded_deref(
        &mut self,
        current: BasicBlockId,
        place: Place,
        source_ty: Ty,
        target_ty: Ty,
        span: MirSpan,
        mutability: bool,
    ) -> Result<Option<(Place, BasicBlockId)>> {
        let (chalk_mut, trait_lang_item, trait_method_name, borrow_kind) = if !mutability {
            (Mutability::Not, LangItem::Deref, name![deref], BorrowKind::Shared)
        } else {
            (
                Mutability::Mut,
                LangItem::DerefMut,
                name![deref_mut],
                BorrowKind::Mut { allow_two_phase_borrow: false },
            )
        };
        let ty_ref = TyKind::Ref(chalk_mut, static_lifetime(), source_ty.clone()).intern(Interner);
        let target_ty_ref = TyKind::Ref(chalk_mut, static_lifetime(), target_ty).intern(Interner);
        let ref_place: Place = self.temp(ty_ref, current, span)?.into();
        self.push_assignment(current, ref_place.clone(), Rvalue::Ref(borrow_kind, place), span);
        let deref_trait = self
            .resolve_lang_item(trait_lang_item)?
            .as_trait()
            .ok_or(MirLowerError::LangItemNotFound(trait_lang_item))?;
        let deref_fn = self
            .db
            .trait_data(deref_trait)
            .method_by_name(&trait_method_name)
            .ok_or(MirLowerError::LangItemNotFound(trait_lang_item))?;
        let deref_fn_op = Operand::const_zst(
            TyKind::FnDef(
                self.db.intern_callable_def(CallableDefId::FunctionId(deref_fn)).into(),
                Substitution::from1(Interner, source_ty),
            )
            .intern(Interner),
        );
        let mut result: Place = self.temp(target_ty_ref, current, span)?.into();
        let Some(current) = self.lower_call(deref_fn_op, Box::new([Operand::Copy(ref_place)]), result.clone(), current, false, span)? else {
            return Ok(None);
        };
        result = result.project(ProjectionElem::Deref);
        Ok(Some((result, current)))
    }
}
