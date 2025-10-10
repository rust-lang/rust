//! MIR lowering for patterns

use hir_def::{AssocItemId, hir::ExprId, signatures::VariantFields};

use crate::{
    BindingMode,
    mir::{
        LocalId, MutBorrowKind, Operand, OperandKind,
        lower::{
            BasicBlockId, BinOp, BindingId, BorrowKind, Either, Expr, FieldId, Idx, Interner,
            MemoryMap, MirLowerCtx, MirLowerError, MirSpan, Mutability, Pat, PatId, Place,
            PlaceElem, ProjectionElem, RecordFieldPat, ResolveValueResult, Result, Rvalue,
            Substitution, SwitchTargets, TerminatorKind, TupleFieldId, TupleId, TyBuilder, TyKind,
            ValueNs, VariantId,
        },
    },
};

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirLowerError::NotSupported(format!($x)))
    };
}

pub(super) enum AdtPatternShape<'a> {
    Tuple { args: &'a [PatId], ellipsis: Option<u32> },
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
    /// Assume that this pattern matches, assign to existing variables.
    Assign,
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
            cond_place,
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

    pub(super) fn pattern_match_assignment(
        &mut self,
        current: BasicBlockId,
        value: Place,
        pattern: PatId,
    ) -> Result<BasicBlockId> {
        let (current, _) =
            self.pattern_match_inner(current, None, value, pattern, MatchingMode::Assign)?;
        Ok(current)
    }

    pub(super) fn match_self_param(
        &mut self,
        id: BindingId,
        current: BasicBlockId,
        local: LocalId,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        self.pattern_match_binding(
            id,
            BindingMode::Move,
            local.into(),
            MirSpan::SelfParam,
            current,
            None,
        )
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
        cond_place.projection = self.result.projection_store.intern(
            cond_place
                .projection
                .lookup(&self.result.projection_store)
                .iter()
                .cloned()
                .chain((0..cnt).map(|_| ProjectionElem::Deref))
                .collect::<Vec<_>>()
                .into(),
        );
        Ok(match &self.body[pattern] {
            Pat::Missing => return Err(MirLowerError::IncompletePattern),
            Pat::Wild => (current, current_else),
            Pat::Tuple { args, ellipsis } => {
                let subst = match self.infer[pattern].kind(Interner) {
                    TyKind::Tuple(_, s) => s,
                    _ => {
                        return Err(MirLowerError::TypeError(
                            "non tuple type matched with tuple pattern",
                        ));
                    }
                };
                self.pattern_match_tuple_like(
                    current,
                    current_else,
                    args,
                    *ellipsis,
                    (0..subst.len(Interner)).map(|i| {
                        PlaceElem::Field(Either::Right(TupleFieldId {
                            tuple: TupleId(!0), // Dummy as it is unused
                            index: i as u32,
                        }))
                    }),
                    &cond_place,
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
                        cond_place,
                        *pat,
                        MatchingMode::Check,
                    )?;
                    if mode != MatchingMode::Check {
                        (next, _) = self.pattern_match_inner(next, None, cond_place, *pat, mode)?;
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
                    if mode == MatchingMode::Check {
                        let ce = *current_else.get_or_insert_with(|| self.new_basic_block());
                        self.set_goto(current, ce, pattern.into());
                    } else {
                        self.set_terminator(current, TerminatorKind::Unreachable, pattern.into());
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
                    AdtPatternShape::Record { args },
                    mode,
                )?
            }
            Pat::Range { start, end } => {
                let mut add_check = |l: &ExprId, binop| -> Result<()> {
                    let lv =
                        self.lower_literal_or_const_to_operand(self.infer[pattern].clone(), l)?;
                    let else_target = *current_else.get_or_insert_with(|| self.new_basic_block());
                    let next = self.new_basic_block();
                    let discr: Place =
                        self.temp(TyBuilder::bool(), current, pattern.into())?.into();
                    self.push_assignment(
                        current,
                        discr,
                        Rvalue::CheckedBinaryOp(
                            binop,
                            lv,
                            Operand { kind: OperandKind::Copy(cond_place), span: None },
                        ),
                        pattern.into(),
                    );
                    let discr = Operand { kind: OperandKind::Copy(discr), span: None };
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
                            place_len,
                            Rvalue::Len(cond_place),
                            pattern.into(),
                        );
                        let else_target =
                            *current_else.get_or_insert_with(|| self.new_basic_block());
                        let next = self.new_basic_block();
                        if slice.is_none() {
                            self.set_terminator(
                                current,
                                TerminatorKind::SwitchInt {
                                    discr: Operand {
                                        kind: OperandKind::Copy(place_len),
                                        span: None,
                                    },
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
                                pattern_len.to_le_bytes().into(),
                                MemoryMap::default(),
                                TyBuilder::usize(),
                            );
                            let discr: Place =
                                self.temp(TyBuilder::bool(), current, pattern.into())?.into();
                            self.push_assignment(
                                current,
                                discr,
                                Rvalue::CheckedBinaryOp(
                                    BinOp::Le,
                                    c,
                                    Operand { kind: OperandKind::Copy(place_len), span: None },
                                ),
                                pattern.into(),
                            );
                            let discr = Operand { kind: OperandKind::Copy(discr), span: None };
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
                    let next_place = cond_place.project(
                        ProjectionElem::ConstantIndex { offset: i as u64, from_end: false },
                        &mut self.result.projection_store,
                    );
                    (current, current_else) =
                        self.pattern_match_inner(current, current_else, next_place, pat, mode)?;
                }
                if let &Some(slice) = slice
                    && mode != MatchingMode::Check
                    && let Pat::Bind { id, subpat: _ } = self.body[slice]
                {
                    let next_place = cond_place.project(
                        ProjectionElem::Subslice {
                            from: prefix.len() as u64,
                            to: suffix.len() as u64,
                        },
                        &mut self.result.projection_store,
                    );
                    let mode = self.infer.binding_modes[slice];
                    (current, current_else) = self.pattern_match_binding(
                        id,
                        mode,
                        next_place,
                        (slice).into(),
                        current,
                        current_else,
                    )?;
                }
                for (i, &pat) in suffix.iter().enumerate() {
                    let next_place = cond_place.project(
                        ProjectionElem::ConstantIndex { offset: i as u64, from_end: true },
                        &mut self.result.projection_store,
                    );
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
                    let unresolved_name = || {
                        MirLowerError::unresolved_path(self.db, p, self.display_target(), self.body)
                    };
                    let hygiene = self.body.pat_path_hygiene(pattern);
                    let pr = self
                        .resolver
                        .resolve_path_in_value_ns(self.db, p, hygiene)
                        .ok_or_else(unresolved_name)?;

                    if let (
                        MatchingMode::Assign,
                        ResolveValueResult::ValueNs(ValueNs::LocalBinding(binding), _),
                    ) = (mode, &pr)
                    {
                        let local = self.binding_local(*binding)?;
                        self.push_match_assignment(
                            current,
                            local,
                            BindingMode::Move,
                            cond_place,
                            pattern.into(),
                        );
                        return Ok((current, current_else));
                    }

                    // The path is not a variant or a local, so it is a const
                    if mode != MatchingMode::Check {
                        // A const don't bind anything. Only needs check.
                        return Ok((current, current_else));
                    }
                    let (c, subst) = 'b: {
                        if let Some(x) = self.infer.assoc_resolutions_for_pat(pattern)
                            && let AssocItemId::ConstId(c) = x.0
                        {
                            break 'b (c, x.1);
                        }
                        if let ResolveValueResult::ValueNs(ValueNs::ConstId(c), _) = pr {
                            break 'b (c, Substitution::empty(Interner));
                        }
                        not_supported!("path in pattern position that is not const or variant")
                    };
                    let tmp: Place =
                        self.temp(self.infer[pattern].clone(), current, pattern.into())?.into();
                    let span = pattern.into();
                    self.lower_const(
                        c.into(),
                        current,
                        tmp,
                        subst,
                        span,
                        self.infer[pattern].clone(),
                    )?;
                    let tmp2: Place = self.temp(TyBuilder::bool(), current, pattern.into())?.into();
                    self.push_assignment(
                        current,
                        tmp2,
                        Rvalue::CheckedBinaryOp(
                            BinOp::Eq,
                            Operand { kind: OperandKind::Copy(tmp), span: None },
                            Operand { kind: OperandKind::Copy(cond_place), span: None },
                        ),
                        span,
                    );
                    let next = self.new_basic_block();
                    let else_target = current_else.unwrap_or_else(|| self.new_basic_block());
                    self.set_terminator(
                        current,
                        TerminatorKind::SwitchInt {
                            discr: Operand { kind: OperandKind::Copy(tmp2), span: None },
                            targets: SwitchTargets::static_if(1, next, else_target),
                        },
                        span,
                    );
                    (next, Some(else_target))
                }
            },
            Pat::Lit(l) => match &self.body[*l] {
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
                    (current, current_else) =
                        self.pattern_match_inner(current, current_else, cond_place, *subpat, mode)?
                }
                if mode != MatchingMode::Check {
                    let mode = self.infer.binding_modes[pattern];
                    self.pattern_match_binding(
                        *id,
                        mode,
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
            Pat::Ref { pat, mutability: _ } => {
                let cond_place =
                    cond_place.project(ProjectionElem::Deref, &mut self.result.projection_store);
                self.pattern_match_inner(current, current_else, cond_place, *pat, mode)?
            }
            &Pat::Expr(expr) => {
                stdx::always!(
                    mode == MatchingMode::Assign,
                    "Pat::Expr can only come in destructuring assignments"
                );
                let Some((lhs_place, current)) = self.lower_expr_as_place(current, expr, false)?
                else {
                    return Ok((current, current_else));
                };
                self.push_assignment(
                    current,
                    lhs_place,
                    Operand { kind: OperandKind::Copy(cond_place), span: None }.into(),
                    expr.into(),
                );
                (current, current_else)
            }
            Pat::Box { .. } => not_supported!("box pattern"),
            Pat::ConstBlock(_) => not_supported!("const block pattern"),
        })
    }

    fn pattern_match_binding(
        &mut self,
        id: BindingId,
        mode: BindingMode,
        cond_place: Place,
        span: MirSpan,
        current: BasicBlockId,
        current_else: Option<BasicBlockId>,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        let target_place = self.binding_local(id)?;
        self.push_storage_live(id, current)?;
        self.push_match_assignment(current, target_place, mode, cond_place, span);
        Ok((current, current_else))
    }

    fn push_match_assignment(
        &mut self,
        current: BasicBlockId,
        target_place: LocalId,
        mode: BindingMode,
        cond_place: Place,
        span: MirSpan,
    ) {
        self.push_assignment(
            current,
            target_place.into(),
            match mode {
                BindingMode::Move => {
                    Operand { kind: OperandKind::Copy(cond_place), span: None }.into()
                }
                BindingMode::Ref(Mutability::Not) => Rvalue::Ref(BorrowKind::Shared, cond_place),
                BindingMode::Ref(Mutability::Mut) => {
                    Rvalue::Ref(BorrowKind::Mut { kind: MutBorrowKind::Default }, cond_place)
                }
            },
            span,
        );
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
            discr,
            Rvalue::CheckedBinaryOp(
                BinOp::Eq,
                c,
                Operand { kind: OperandKind::Copy(cond_place), span: None },
            ),
            pattern.into(),
        );
        let discr = Operand { kind: OperandKind::Copy(discr), span: None };
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
                    self.push_assignment(current, tmp, Rvalue::Discriminant(cond_place), span);
                    let next = self.new_basic_block();
                    let else_target = current_else.get_or_insert_with(|| self.new_basic_block());
                    self.set_terminator(
                        current,
                        TerminatorKind::SwitchInt {
                            discr: Operand { kind: OperandKind::Copy(tmp), span: None },
                            targets: SwitchTargets::static_if(e, next, *else_target),
                        },
                        span,
                    );
                    current = next;
                }
                self.pattern_matching_variant_fields(
                    shape,
                    v.fields(self.db),
                    variant,
                    current,
                    current_else,
                    &cond_place,
                    mode,
                )?
            }
            VariantId::StructId(s) => self.pattern_matching_variant_fields(
                shape,
                s.fields(self.db),
                variant,
                current,
                current_else,
                &cond_place,
                mode,
            )?,
            VariantId::UnionId(_) => {
                return Err(MirLowerError::TypeError("pattern matching on union"));
            }
        })
    }

    fn pattern_matching_variant_fields(
        &mut self,
        shape: AdtPatternShape<'_>,
        variant_data: &VariantFields,
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
                            PlaceElem::Field(Either::Left(FieldId {
                                parent: v,
                                local_id: field_id,
                            })),
                            x.pat,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.pattern_match_adt(current, current_else, it.into_iter(), cond_place, mode)?
            }
            AdtPatternShape::Tuple { args, ellipsis } => {
                let fields = variant_data.fields().iter().map(|(x, _)| {
                    PlaceElem::Field(Either::Left(FieldId { parent: v, local_id: x }))
                });
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
            let cond_place = cond_place.project(proj, &mut self.result.projection_store);
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
        ellipsis: Option<u32>,
        fields: impl DoubleEndedIterator<Item = PlaceElem> + Clone,
        cond_place: &Place,
        mode: MatchingMode,
    ) -> Result<(BasicBlockId, Option<BasicBlockId>)> {
        let (al, ar) = args.split_at(ellipsis.map_or(args.len(), |it| it as usize));
        let it = al
            .iter()
            .zip(fields.clone())
            .chain(ar.iter().rev().zip(fields.rev()))
            .map(|(x, y)| (y, *x));
        self.pattern_match_adt(current, current_else, it, cond_place, mode)
    }
}
