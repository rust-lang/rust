//! MIR lowering for patterns

use crate::utils::pattern_matching_dereference_count;

use super::*;

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirLowerError::NotSupported(format!($x)))
    };
}

pub(super) enum AdtPatternShape<'a> {
    Tuple { args: &'a [PatId], ellipsis: Option<usize> },
    Record { args: &'a [RecordFieldPat] },
    Unit,
}

impl MirLowerCtx<'_> {
    /// It gets a `current` unterminated block, appends some statements and possibly a terminator to it to check if
    /// the pattern matches and write bindings, and returns two unterminated blocks, one for the matched path (which
    /// can be the `current` block) and one for the mismatched path. If the input pattern is irrefutable, the
    /// mismatched path block is `None`.
    ///
    /// By default, it will create a new block for mismatched path. If you already have one, you can provide it with
    /// `current_else` argument to save an unnecessary jump. If `current_else` isn't `None`, the result mismatched path
    /// wouldn't be `None` as well. Note that this function will add jumps to the beginning of the `current_else` block,
    /// so it should be an empty block.
    pub(super) fn pattern_match(
        &mut self,
        mut current: BasicBlockId,
        mut current_else: Option<BasicBlockId>,
        mut cond_place: Place,
        mut cond_ty: Ty,
        pattern: PatId,
        mut binding_mode: BindingAnnotation,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        Ok(match &self.body.pats[pattern] {
            Pat::Missing => return Err(MirLowerError::IncompleteExpr),
            Pat::Wild => (current, current_else),
            Pat::Tuple { args, ellipsis } => {
                pattern_matching_dereference(&mut cond_ty, &mut binding_mode, &mut cond_place);
                let subst = match cond_ty.kind(Interner) {
                    TyKind::Tuple(_, s) => s,
                    _ => {
                        return Err(MirLowerError::TypeError(
                            "non tuple type matched with tuple pattern",
                        ))
                    }
                };
                self.pattern_match_tuple_like(
                    current,
                    current_else,
                    args,
                    *ellipsis,
                    subst.iter(Interner).enumerate().map(|(i, x)| {
                        (PlaceElem::TupleOrClosureField(i), x.assert_ty_ref(Interner).clone())
                    }),
                    &cond_place,
                    binding_mode,
                )?
            }
            Pat::Or(pats) => {
                let then_target = self.new_basic_block();
                let mut finished = false;
                for pat in &**pats {
                    let (next, next_else) = self.pattern_match(
                        current,
                        None,
                        cond_place.clone(),
                        cond_ty.clone(),
                        *pat,
                        binding_mode,
                    )?;
                    self.set_goto(next, then_target);
                    match next_else {
                        Some(t) => {
                            current = t;
                        }
                        None => {
                            finished = true;
                            break;
                        }
                    }
                }
                if !finished {
                    let ce = *current_else.get_or_insert_with(|| self.new_basic_block());
                    self.set_goto(current, ce);
                }
                (then_target, current_else)
            }
            Pat::Record { args, .. } => {
                let Some(variant) = self.infer.variant_resolution_for_pat(pattern) else {
                    not_supported!("unresolved variant");
                };
                self.pattern_matching_variant(
                    cond_ty,
                    binding_mode,
                    cond_place,
                    variant,
                    current,
                    pattern.into(),
                    current_else,
                    AdtPatternShape::Record { args: &*args },
                )?
            }
            Pat::Range { .. } => not_supported!("range pattern"),
            Pat::Slice { .. } => not_supported!("slice pattern"),
            Pat::Path(_) => {
                let Some(variant) = self.infer.variant_resolution_for_pat(pattern) else {
                    not_supported!("unresolved variant");
                };
                self.pattern_matching_variant(
                    cond_ty,
                    binding_mode,
                    cond_place,
                    variant,
                    current,
                    pattern.into(),
                    current_else,
                    AdtPatternShape::Unit,
                )?
            }
            Pat::Lit(l) => match &self.body.exprs[*l] {
                Expr::Literal(l) => {
                    let c = self.lower_literal_to_operand(cond_ty, l)?;
                    self.pattern_match_const(current_else, current, c, cond_place, pattern)?
                }
                _ => not_supported!("expression path literal"),
            },
            Pat::Bind { id, subpat } => {
                let target_place = self.result.binding_locals[*id];
                let mode = self.body.bindings[*id].mode;
                if let Some(subpat) = subpat {
                    (current, current_else) = self.pattern_match(
                        current,
                        current_else,
                        cond_place.clone(),
                        cond_ty,
                        *subpat,
                        binding_mode,
                    )?
                }
                if matches!(mode, BindingAnnotation::Ref | BindingAnnotation::RefMut) {
                    binding_mode = mode;
                }
                self.push_storage_live(*id, current)?;
                self.push_assignment(
                    current,
                    target_place.into(),
                    match binding_mode {
                        BindingAnnotation::Unannotated | BindingAnnotation::Mutable => {
                            Operand::Copy(cond_place).into()
                        }
                        BindingAnnotation::Ref => Rvalue::Ref(BorrowKind::Shared, cond_place),
                        BindingAnnotation::RefMut => Rvalue::Ref(
                            BorrowKind::Mut { allow_two_phase_borrow: false },
                            cond_place,
                        ),
                    },
                    pattern.into(),
                );
                (current, current_else)
            }
            Pat::TupleStruct { path: _, args, ellipsis } => {
                let Some(variant) = self.infer.variant_resolution_for_pat(pattern) else {
                    not_supported!("unresolved variant");
                };
                self.pattern_matching_variant(
                    cond_ty,
                    binding_mode,
                    cond_place,
                    variant,
                    current,
                    pattern.into(),
                    current_else,
                    AdtPatternShape::Tuple { args, ellipsis: *ellipsis },
                )?
            }
            Pat::Ref { pat, mutability: _ } => {
                if let Some((ty, _, _)) = cond_ty.as_reference() {
                    cond_ty = ty.clone();
                    cond_place.projection.push(ProjectionElem::Deref);
                    self.pattern_match(
                        current,
                        current_else,
                        cond_place,
                        cond_ty,
                        *pat,
                        binding_mode,
                    )?
                } else {
                    return Err(MirLowerError::TypeError("& pattern for non reference"));
                }
            }
            Pat::Box { .. } => not_supported!("box pattern"),
            Pat::ConstBlock(_) => not_supported!("const block pattern"),
        })
    }

    fn pattern_match_const(
        &mut self,
        current_else: Option<BasicBlockId>,
        current: BasicBlockId,
        c: Operand,
        cond_place: Place,
        pattern: Idx<Pat>,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        let then_target = self.new_basic_block();
        let else_target = current_else.unwrap_or_else(|| self.new_basic_block());
        let discr: Place = self.temp(TyBuilder::bool())?.into();
        self.push_assignment(
            current,
            discr.clone(),
            Rvalue::CheckedBinaryOp(BinOp::Eq, c, Operand::Copy(cond_place)),
            pattern.into(),
        );
        let discr = Operand::Copy(discr);
        self.set_terminator(
            current,
            Terminator::SwitchInt {
                discr,
                targets: SwitchTargets::static_if(1, then_target, else_target),
            },
        );
        Ok((then_target, Some(else_target)))
    }

    pub(super) fn pattern_matching_variant(
        &mut self,
        mut cond_ty: Ty,
        mut binding_mode: BindingAnnotation,
        mut cond_place: Place,
        variant: VariantId,
        current: BasicBlockId,
        span: MirSpan,
        current_else: Option<BasicBlockId>,
        shape: AdtPatternShape<'_>,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        pattern_matching_dereference(&mut cond_ty, &mut binding_mode, &mut cond_place);
        let subst = match cond_ty.kind(Interner) {
            TyKind::Adt(_, s) => s,
            _ => return Err(MirLowerError::TypeError("non adt type matched with tuple struct")),
        };
        Ok(match variant {
            VariantId::EnumVariantId(v) => {
                let e = self.db.const_eval_discriminant(v)? as u128;
                let next = self.new_basic_block();
                let tmp = self.discr_temp_place();
                self.push_assignment(
                    current,
                    tmp.clone(),
                    Rvalue::Discriminant(cond_place.clone()),
                    span,
                );
                let else_target = current_else.unwrap_or_else(|| self.new_basic_block());
                self.set_terminator(
                    current,
                    Terminator::SwitchInt {
                        discr: Operand::Copy(tmp),
                        targets: SwitchTargets::static_if(e, next, else_target),
                    },
                );
                let enum_data = self.db.enum_data(v.parent);
                self.pattern_matching_variant_fields(
                    shape,
                    &enum_data.variants[v.local_id].variant_data,
                    variant,
                    subst,
                    next,
                    Some(else_target),
                    &cond_place,
                    binding_mode,
                )?
            }
            VariantId::StructId(s) => {
                let struct_data = self.db.struct_data(s);
                self.pattern_matching_variant_fields(
                    shape,
                    &struct_data.variant_data,
                    variant,
                    subst,
                    current,
                    current_else,
                    &cond_place,
                    binding_mode,
                )?
            }
            VariantId::UnionId(_) => {
                return Err(MirLowerError::TypeError("pattern matching on union"))
            }
        })
    }

    fn pattern_matching_variant_fields(
        &mut self,
        shape: AdtPatternShape<'_>,
        variant_data: &VariantData,
        v: VariantId,
        subst: &Substitution,
        current: BasicBlockId,
        current_else: Option<BasicBlockId>,
        cond_place: &Place,
        binding_mode: BindingAnnotation,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        let fields_type = self.db.field_types(v);
        Ok(match shape {
            AdtPatternShape::Record { args } => {
                let it = args
                    .iter()
                    .map(|x| {
                        let field_id =
                            variant_data.field(&x.name).ok_or(MirLowerError::UnresolvedField)?;
                        Ok((
                            PlaceElem::Field(FieldId { parent: v.into(), local_id: field_id }),
                            x.pat,
                            fields_type[field_id].clone().substitute(Interner, subst),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.pattern_match_adt(
                    current,
                    current_else,
                    it.into_iter(),
                    cond_place,
                    binding_mode,
                )?
            }
            AdtPatternShape::Tuple { args, ellipsis } => {
                let fields = variant_data.fields().iter().map(|(x, _)| {
                    (
                        PlaceElem::Field(FieldId { parent: v.into(), local_id: x }),
                        fields_type[x].clone().substitute(Interner, subst),
                    )
                });
                self.pattern_match_tuple_like(
                    current,
                    current_else,
                    args,
                    ellipsis,
                    fields,
                    cond_place,
                    binding_mode,
                )?
            }
            AdtPatternShape::Unit => (current, current_else),
        })
    }

    fn pattern_match_adt(
        &mut self,
        mut current: BasicBlockId,
        mut current_else: Option<BasicBlockId>,
        args: impl Iterator<Item = (PlaceElem, PatId, Ty)>,
        cond_place: &Place,
        binding_mode: BindingAnnotation,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        for (proj, arg, ty) in args {
            let mut cond_place = cond_place.clone();
            cond_place.projection.push(proj);
            (current, current_else) =
                self.pattern_match(current, current_else, cond_place, ty, arg, binding_mode)?;
        }
        Ok((current, current_else))
    }

    fn pattern_match_tuple_like(
        &mut self,
        current: BasicBlockId,
        current_else: Option<BasicBlockId>,
        args: &[PatId],
        ellipsis: Option<usize>,
        fields: impl DoubleEndedIterator<Item = (PlaceElem, Ty)> + Clone,
        cond_place: &Place,
        binding_mode: BindingAnnotation,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        let (al, ar) = args.split_at(ellipsis.unwrap_or(args.len()));
        let it = al
            .iter()
            .zip(fields.clone())
            .chain(ar.iter().rev().zip(fields.rev()))
            .map(|(x, y)| (y.0, *x, y.1));
        self.pattern_match_adt(current, current_else, it, cond_place, binding_mode)
    }
}

fn pattern_matching_dereference(
    cond_ty: &mut Ty,
    binding_mode: &mut BindingAnnotation,
    cond_place: &mut Place,
) {
    let cnt = pattern_matching_dereference_count(cond_ty, binding_mode);
    cond_place.projection.extend((0..cnt).map(|_| ProjectionElem::Deref));
}
