//! MIR lowering for patterns

use hir_def::{hir::LiteralOrConst, resolver::HasResolver, AssocItemId};

use crate::BindingMode;

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

/// We need to do pattern matching in two phases: One to check if the pattern matches, and one to fill the bindings
/// of patterns. This is necessary to prevent double moves and similar problems. For example:
/// ```ignore
/// struct X;
/// match (X, 3) {
///     (b, 2) | (b, 3) => {},
///     _ => {}
/// }
/// ```
/// If we do everything in one pass, we will move `X` to the first `b`, then we see that the second field of tuple
/// doesn't match and we should move the `X` to the second `b` (which here is the same thing, but doesn't need to be) and
/// it might even doesn't match the second pattern and we may want to not move `X` at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchingMode {
    /// Check that if this pattern matches
    Check,
    /// Assume that this pattern matches, fill bindings
    Bind,
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
        current: BasicBlockId,
        current_else: Option<BasicBlockId>,
        cond_place: Place,
        pattern: PatId,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        let (current, current_else) = self.pattern_match_inner(
            current,
            current_else,
            cond_place.clone(),
            pattern,
            MatchingMode::Check,
        )?;
        let (current, current_else) = self.pattern_match_inner(
            current,
            current_else,
            cond_place,
            pattern,
            MatchingMode::Bind,
        )?;
        Ok((current, current_else))
    }

    fn pattern_match_inner(
        &mut self,
        mut current: BasicBlockId,
        mut current_else: Option<BasicBlockId>,
        mut cond_place: Place,
        pattern: PatId,
        mode: MatchingMode,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        let cnt = self.infer.pat_adjustments.get(&pattern).map(|x| x.len()).unwrap_or_default();
        cond_place.projection = cond_place
            .projection
            .iter()
            .cloned()
            .chain((0..cnt).map(|_| ProjectionElem::Deref))
            .collect::<Vec<_>>()
            .into();
        Ok(match &self.body.pats[pattern] {
            Pat::Missing => return Err(MirLowerError::IncompletePattern),
            Pat::Wild => (current, current_else),
            Pat::Tuple { args, ellipsis } => {
                let subst = match self.infer[pattern].kind(Interner) {
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
                    (0..subst.len(Interner)).map(|i| PlaceElem::TupleOrClosureField(i)),
                    &(&mut cond_place),
                    mode,
                )?
            }
            Pat::Or(pats) => {
                let then_target = self.new_basic_block();
                let mut finished = false;
                for pat in &**pats {
                    let (mut next, next_else) = self.pattern_match_inner(
                        current,
                        None,
                        (&mut cond_place).clone(),
                        *pat,
                        MatchingMode::Check,
                    )?;
                    if mode == MatchingMode::Bind {
                        (next, _) = self.pattern_match_inner(
                            next,
                            None,
                            (&mut cond_place).clone(),
                            *pat,
                            MatchingMode::Bind,
                        )?;
                    }
                    self.set_goto(next, then_target, pattern.into());
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
                    if mode == MatchingMode::Bind {
                        self.set_terminator(current, TerminatorKind::Unreachable, pattern.into());
                    } else {
                        let ce = *current_else.get_or_insert_with(|| self.new_basic_block());
                        self.set_goto(current, ce, pattern.into());
                    }
                }
                (then_target, current_else)
            }
            Pat::Record { args, .. } => {
                let Some(variant) = self.infer.variant_resolution_for_pat(pattern) else {
                    not_supported!("unresolved variant for record");
                };
                self.pattern_matching_variant(
                    cond_place,
                    variant,
                    current,
                    pattern.into(),
                    current_else,
                    AdtPatternShape::Record { args: &*args },
                    mode,
                )?
            }
            Pat::Range { start, end } => {
                let mut add_check = |l: &LiteralOrConst, binop| -> Result<()> {
                    let lv =
                        self.lower_literal_or_const_to_operand(self.infer[pattern].clone(), l)?;
                    let else_target = *current_else.get_or_insert_with(|| self.new_basic_block());
                    let next = self.new_basic_block();
                    let discr: Place =
                        self.temp(TyBuilder::bool(), current, pattern.into())?.into();
                    self.push_assignment(
                        current,
                        discr.clone(),
                        Rvalue::CheckedBinaryOp(
                            binop,
                            lv,
                            Operand::Copy((&mut cond_place).clone()),
                        ),
                        pattern.into(),
                    );
                    let discr = Operand::Copy(discr);
                    self.set_terminator(
                        current,
                        TerminatorKind::SwitchInt {
                            discr,
                            targets: SwitchTargets::static_if(1, next, else_target),
                        },
                        pattern.into(),
                    );
                    current = next;
                    Ok(())
                };
                if mode == MatchingMode::Check {
                    if let Some(start) = start {
                        add_check(start, BinOp::Le)?;
                    }
                    if let Some(end) = end {
                        add_check(end, BinOp::Ge)?;
                    }
                }
                (current, current_else)
            }
            Pat::Slice { prefix, slice, suffix } => {
                if mode == MatchingMode::Check {
                    // emit runtime length check for slice
                    if let TyKind::Slice(_) = self.infer[pattern].kind(Interner) {
                        let pattern_len = prefix.len() + suffix.len();
                        let place_len: Place =
                            self.temp(TyBuilder::usize(), current, pattern.into())?.into();
                        self.push_assignment(
                            current,
                            place_len.clone(),
                            Rvalue::Len((&mut cond_place).clone()),
                            pattern.into(),
                        );
                        let else_target =
                            *current_else.get_or_insert_with(|| self.new_basic_block());
                        let next = self.new_basic_block();
                        if slice.is_none() {
                            self.set_terminator(
                                current,
                                TerminatorKind::SwitchInt {
                                    discr: Operand::Copy(place_len),
                                    targets: SwitchTargets::static_if(
                                        pattern_len as u128,
                                        next,
                                        else_target,
                                    ),
                                },
                                pattern.into(),
                            );
                        } else {
                            let c = Operand::from_concrete_const(
                                pattern_len.to_le_bytes().to_vec(),
                                MemoryMap::default(),
                                TyBuilder::usize(),
                            );
                            let discr: Place =
                                self.temp(TyBuilder::bool(), current, pattern.into())?.into();
                            self.push_assignment(
                                current,
                                discr.clone(),
                                Rvalue::CheckedBinaryOp(BinOp::Le, c, Operand::Copy(place_len)),
                                pattern.into(),
                            );
                            let discr = Operand::Copy(discr);
                            self.set_terminator(
                                current,
                                TerminatorKind::SwitchInt {
                                    discr,
                                    targets: SwitchTargets::static_if(1, next, else_target),
                                },
                                pattern.into(),
                            );
                        }
                        current = next;
                    }
                }
                for (i, &pat) in prefix.iter().enumerate() {
                    let next_place = (&mut cond_place).project(ProjectionElem::ConstantIndex {
                        offset: i as u64,
                        from_end: false,
                    });
                    (current, current_else) =
                        self.pattern_match_inner(current, current_else, next_place, pat, mode)?;
                }
                if let Some(slice) = slice {
                    if mode == MatchingMode::Bind {
                        if let Pat::Bind { id, subpat: _ } = self.body[*slice] {
                            let next_place = (&mut cond_place).project(ProjectionElem::Subslice {
                                from: prefix.len() as u64,
                                to: suffix.len() as u64,
                            });
                            (current, current_else) = self.pattern_match_binding(
                                id,
                                next_place,
                                (*slice).into(),
                                current,
                                current_else,
                            )?;
                        }
                    }
                }
                for (i, &pat) in suffix.iter().enumerate() {
                    let next_place = (&mut cond_place).project(ProjectionElem::ConstantIndex {
                        offset: i as u64,
                        from_end: true,
                    });
                    (current, current_else) =
                        self.pattern_match_inner(current, current_else, next_place, pat, mode)?;
                }
                (current, current_else)
            }
            Pat::Path(p) => match self.infer.variant_resolution_for_pat(pattern) {
                Some(variant) => self.pattern_matching_variant(
                    cond_place,
                    variant,
                    current,
                    pattern.into(),
                    current_else,
                    AdtPatternShape::Unit,
                    mode,
                )?,
                None => {
                    // The path is not a variant, so it is a const
                    if mode != MatchingMode::Check {
                        // A const don't bind anything. Only needs check.
                        return Ok((current, current_else));
                    }
                    let unresolved_name = || MirLowerError::unresolved_path(self.db, p);
                    let resolver = self.owner.resolver(self.db.upcast());
                    let pr = resolver
                        .resolve_path_in_value_ns(self.db.upcast(), p)
                        .ok_or_else(unresolved_name)?;
                    let (c, subst) = 'b: {
                        if let Some(x) = self.infer.assoc_resolutions_for_pat(pattern) {
                            if let AssocItemId::ConstId(c) = x.0 {
                                break 'b (c, x.1);
                            }
                        }
                        if let ResolveValueResult::ValueNs(v) = pr {
                            if let ValueNs::ConstId(c) = v {
                                break 'b (c, Substitution::empty(Interner));
                            }
                        }
                        not_supported!("path in pattern position that is not const or variant")
                    };
                    let tmp: Place =
                        self.temp(self.infer[pattern].clone(), current, pattern.into())?.into();
                    let span = pattern.into();
                    self.lower_const(
                        c.into(),
                        current,
                        tmp.clone(),
                        subst,
                        span,
                        self.infer[pattern].clone(),
                    )?;
                    let tmp2: Place = self.temp(TyBuilder::bool(), current, pattern.into())?.into();
                    self.push_assignment(
                        current,
                        tmp2.clone(),
                        Rvalue::CheckedBinaryOp(
                            BinOp::Eq,
                            Operand::Copy(tmp),
                            Operand::Copy(cond_place),
                        ),
                        span,
                    );
                    let next = self.new_basic_block();
                    let else_target = current_else.unwrap_or_else(|| self.new_basic_block());
                    self.set_terminator(
                        current,
                        TerminatorKind::SwitchInt {
                            discr: Operand::Copy(tmp2),
                            targets: SwitchTargets::static_if(1, next, else_target),
                        },
                        span,
                    );
                    (next, Some(else_target))
                }
            },
            Pat::Lit(l) => match &self.body.exprs[*l] {
                Expr::Literal(l) => {
                    if mode == MatchingMode::Check {
                        let c = self.lower_literal_to_operand(self.infer[pattern].clone(), l)?;
                        self.pattern_match_const(current_else, current, c, cond_place, pattern)?
                    } else {
                        (current, current_else)
                    }
                }
                _ => not_supported!("expression path literal"),
            },
            Pat::Bind { id, subpat } => {
                if let Some(subpat) = subpat {
                    (current, current_else) = self.pattern_match_inner(
                        current,
                        current_else,
                        (&mut cond_place).clone(),
                        *subpat,
                        mode,
                    )?
                }
                if mode == MatchingMode::Bind {
                    self.pattern_match_binding(
                        *id,
                        cond_place,
                        pattern.into(),
                        current,
                        current_else,
                    )?
                } else {
                    (current, current_else)
                }
            }
            Pat::TupleStruct { path: _, args, ellipsis } => {
                let Some(variant) = self.infer.variant_resolution_for_pat(pattern) else {
                    not_supported!("unresolved variant");
                };
                self.pattern_matching_variant(
                    cond_place,
                    variant,
                    current,
                    pattern.into(),
                    current_else,
                    AdtPatternShape::Tuple { args, ellipsis: *ellipsis },
                    mode,
                )?
            }
            Pat::Ref { pat, mutability: _ } => self.pattern_match_inner(
                current,
                current_else,
                cond_place.project(ProjectionElem::Deref),
                *pat,
                mode,
            )?,
            Pat::Box { .. } => not_supported!("box pattern"),
            Pat::ConstBlock(_) => not_supported!("const block pattern"),
        })
    }

    fn pattern_match_binding(
        &mut self,
        id: BindingId,
        cond_place: Place,
        span: MirSpan,
        current: BasicBlockId,
        current_else: Option<BasicBlockId>,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        let target_place = self.binding_local(id)?;
        let mode = self.infer.binding_modes[id];
        self.push_storage_live(id, current)?;
        self.push_assignment(
            current,
            target_place.into(),
            match mode {
                BindingMode::Move => Operand::Copy(cond_place).into(),
                BindingMode::Ref(Mutability::Not) => Rvalue::Ref(BorrowKind::Shared, cond_place),
                BindingMode::Ref(Mutability::Mut) => {
                    Rvalue::Ref(BorrowKind::Mut { allow_two_phase_borrow: false }, cond_place)
                }
            },
            span,
        );
        Ok((current, current_else))
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
        let discr: Place = self.temp(TyBuilder::bool(), current, pattern.into())?.into();
        self.push_assignment(
            current,
            discr.clone(),
            Rvalue::CheckedBinaryOp(BinOp::Eq, c, Operand::Copy(cond_place)),
            pattern.into(),
        );
        let discr = Operand::Copy(discr);
        self.set_terminator(
            current,
            TerminatorKind::SwitchInt {
                discr,
                targets: SwitchTargets::static_if(1, then_target, else_target),
            },
            pattern.into(),
        );
        Ok((then_target, Some(else_target)))
    }

    fn pattern_matching_variant(
        &mut self,
        cond_place: Place,
        variant: VariantId,
        mut current: BasicBlockId,
        span: MirSpan,
        mut current_else: Option<BasicBlockId>,
        shape: AdtPatternShape<'_>,
        mode: MatchingMode,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        Ok(match variant {
            VariantId::EnumVariantId(v) => {
                if mode == MatchingMode::Check {
                    let e = self.const_eval_discriminant(v)? as u128;
                    let tmp = self.discr_temp_place(current);
                    self.push_assignment(
                        current,
                        tmp.clone(),
                        Rvalue::Discriminant(cond_place.clone()),
                        span,
                    );
                    let next = self.new_basic_block();
                    let else_target = current_else.get_or_insert_with(|| self.new_basic_block());
                    self.set_terminator(
                        current,
                        TerminatorKind::SwitchInt {
                            discr: Operand::Copy(tmp),
                            targets: SwitchTargets::static_if(e, next, *else_target),
                        },
                        span,
                    );
                    current = next;
                }
                let enum_data = self.db.enum_data(v.parent);
                self.pattern_matching_variant_fields(
                    shape,
                    &enum_data.variants[v.local_id].variant_data,
                    variant,
                    current,
                    current_else,
                    &cond_place,
                    mode,
                )?
            }
            VariantId::StructId(s) => {
                let struct_data = self.db.struct_data(s);
                self.pattern_matching_variant_fields(
                    shape,
                    &struct_data.variant_data,
                    variant,
                    current,
                    current_else,
                    &cond_place,
                    mode,
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
        current: BasicBlockId,
        current_else: Option<BasicBlockId>,
        cond_place: &Place,
        mode: MatchingMode,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
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
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.pattern_match_adt(current, current_else, it.into_iter(), cond_place, mode)?
            }
            AdtPatternShape::Tuple { args, ellipsis } => {
                let fields = variant_data
                    .fields()
                    .iter()
                    .map(|(x, _)| PlaceElem::Field(FieldId { parent: v.into(), local_id: x }));
                self.pattern_match_tuple_like(
                    current,
                    current_else,
                    args,
                    ellipsis,
                    fields,
                    cond_place,
                    mode,
                )?
            }
            AdtPatternShape::Unit => (current, current_else),
        })
    }

    fn pattern_match_adt(
        &mut self,
        mut current: BasicBlockId,
        mut current_else: Option<BasicBlockId>,
        args: impl Iterator<Item = (PlaceElem, PatId)>,
        cond_place: &Place,
        mode: MatchingMode,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        for (proj, arg) in args {
            let cond_place = cond_place.project(proj);
            (current, current_else) =
                self.pattern_match_inner(current, current_else, cond_place, arg, mode)?;
        }
        Ok((current, current_else))
    }

    fn pattern_match_tuple_like(
        &mut self,
        current: BasicBlockId,
        current_else: Option<BasicBlockId>,
        args: &[PatId],
        ellipsis: Option<usize>,
        fields: impl DoubleEndedIterator<Item = PlaceElem> + Clone,
        cond_place: &Place,
        mode: MatchingMode,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        let (al, ar) = args.split_at(ellipsis.unwrap_or(args.len()));
        let it = al
            .iter()
            .zip(fields.clone())
            .chain(ar.iter().rev().zip(fields.rev()))
            .map(|(x, y)| (y, *x));
        self.pattern_match_adt(current, current_else, it, cond_place, mode)
    }
}
