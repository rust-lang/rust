use either::Either;
use rustc_middle::mir::*;
use rustc_middle::thir::{self, *};
use rustc_middle::ty::{self, Ty, TypeVisitableExt};

use crate::builder::Builder;
use crate::builder::expr::as_place::{PlaceBase, PlaceBuilder};
use crate::builder::matches::util::Range;
use crate::builder::matches::{FlatPat, MatchPairTree, TestCase};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Builds and returns [`MatchPairTree`] subtrees, one for each pattern in
    /// `subpatterns`, representing the fields of a [`PatKind::Variant`] or
    /// [`PatKind::Leaf`].
    ///
    /// Used internally by [`MatchPairTree::for_pattern`].
    fn field_match_pairs<'pat>(
        &mut self,
        place: PlaceBuilder<'tcx>,
        subpatterns: &'pat [FieldPat<'tcx>],
    ) -> Vec<MatchPairTree<'pat, 'tcx>> {
        subpatterns
            .iter()
            .map(|fieldpat| {
                let place =
                    place.clone_project(PlaceElem::Field(fieldpat.field, fieldpat.pattern.ty));
                MatchPairTree::for_pattern(place, &fieldpat.pattern, self)
            })
            .collect()
    }

    /// Builds [`MatchPairTree`] subtrees for the prefix/middle/suffix parts of an
    /// array pattern or slice pattern, and adds those trees to `match_pairs`.
    ///
    /// Used internally by [`MatchPairTree::for_pattern`].
    fn prefix_slice_suffix<'pat>(
        &mut self,
        src_path: &'pat Pat<'tcx>,
        match_pairs: &mut Vec<MatchPairTree<'pat, 'tcx>>,
        place: &PlaceBuilder<'tcx>,
        prefix: &'pat [Box<Pat<'tcx>>],
        opt_slice: &'pat Option<Box<Pat<'tcx>>>,
        suffix: &'pat [Box<Pat<'tcx>>],
    ) {
        let tcx = self.tcx;
        let (min_length, exact_size) = if let Some(place_resolved) = place.try_to_place(self) {
            match place_resolved.ty(&self.local_decls, tcx).ty.kind() {
                ty::Array(_, length) => (
                    length
                        .try_to_target_usize(tcx)
                        .expect("expected len of array pat to be definite"),
                    true,
                ),
                _ => ((prefix.len() + suffix.len()).try_into().unwrap(), false),
            }
        } else {
            ((prefix.len() + suffix.len()).try_into().unwrap(), false)
        };

        if self.should_optimize_subslice(prefix) {
            let elem_ty = prefix[0].ty;
            let prefix_valtree = self.simplify_const_pattern_slice_into_valtree(prefix);
            let match_pair = self.valtree_to_match_pair(
                src_path,
                prefix_valtree,
                place.clone(),
                elem_ty,
                Range::from_start(0..prefix.len() as u64),
                opt_slice.is_some() || !suffix.is_empty(),
            );

            match_pairs.push(match_pair);
        } else {
            match_pairs.extend(prefix.iter().enumerate().map(|(idx, subpattern)| {
                let elem = ProjectionElem::ConstantIndex {
                    offset: idx as u64,
                    min_length,
                    from_end: false,
                };
                MatchPairTree::for_pattern(place.clone_project(elem), subpattern, self)
            }));
        }

        if let Some(subslice_pat) = opt_slice {
            let suffix_len = suffix.len() as u64;
            let subslice = place.clone_project(PlaceElem::Subslice {
                from: prefix.len() as u64,
                to: if exact_size { min_length - suffix_len } else { suffix_len },
                from_end: !exact_size,
            });
            match_pairs.push(MatchPairTree::for_pattern(subslice, subslice_pat, self));
        }

        if self.should_optimize_subslice(suffix) {
            let elem_ty = suffix[0].ty;
            let suffix_valtree = self.simplify_const_pattern_slice_into_valtree(suffix);
            let match_pair = self.valtree_to_match_pair(
                src_path,
                suffix_valtree,
                place.clone(),
                elem_ty,
                Range::from_end(0..suffix.len() as u64),
                true,
            );

            match_pairs.push(match_pair);
        } else {
            match_pairs.extend(suffix.iter().rev().enumerate().map(|(idx, subpattern)| {
                let end_offset = (idx + 1) as u64;
                let elem = ProjectionElem::ConstantIndex {
                    offset: if exact_size { min_length - end_offset } else { end_offset },
                    min_length,
                    from_end: !exact_size,
                };
                let place = place.clone_project(elem);
                MatchPairTree::for_pattern(place, subpattern, self)
            }));
        }
    }

    fn build_slice_branch<'pat>(
        &'pat mut self,
        place: &'pat PlaceBuilder<'tcx>,
        top_pattern: &'pat Pat<'tcx>,
        pattern: &'pat [Box<Pat<'tcx>>],
    ) -> impl Iterator<Item = MatchPairTree<'pat, 'tcx>> + use<'pat, 'tcx, 'a> {
        let entries = self.find_const_groups(pattern);
        let solo = entries.len() == 1;

        let maybe_project = move |base: &PlaceBuilder<'tcx>, elem| {
            if solo { base.clone_project(elem) } else { base.clone() }
        };

        entries.into_iter().map(move |entry| {
            let pattern_len = pattern.len() as u64;
            let mut build_single = |idx| {
                let subpattern = &pattern[idx as usize];
                let place = maybe_project(place, ProjectionElem::ConstantIndex {
                    offset: idx,
                    min_length: pattern_len,
                    from_end: false,
                });

                MatchPairTree::for_pattern(place, subpattern, self)
            };

            match entry {
                Either::Right(range) if range.len() > 1 => {
                    assert!(
                        (range.start..range.end)
                            .all(|idx| self.is_constant_pattern(&pattern[idx as usize]))
                    );

                    let subpattern = &pattern[range.start as usize..range.end as usize];
                    let elem_ty = subpattern[0].ty;

                    let place = maybe_project(place, PlaceElem::Subslice {
                        from: range.start,
                        to: pattern.len() as u64 - range.end,
                        from_end: true,
                    });

                    let valtree = self.simplify_const_pattern_slice_into_valtree(subpattern);
                    self.valtree_to_match_pair(top_pattern, valtree, place, elem_ty, range, false)
                }
                Either::Right(range) => {
                    let tree = build_single(range.start);
                    assert!(self.is_constant_pattern(&pattern[range.start as usize]));
                    tree
                }
                Either::Left(idx) => build_single(idx),
            }
        })
    }

    fn find_const_groups(&self, pattern: &[Box<Pat<'tcx>>]) -> Vec<Either<u64, Range>> {
        let mut entries = Vec::new();
        let mut current_seq_start = None;

        let mut apply = |state: &mut _, idx| {
            if let Some(start) = *state {
                *state = None;
                entries.push(Either::Right(Range::from_start(start..idx)));
            } else {
                entries.push(Either::Left(idx));
            }
        };

        for (idx, pat) in pattern.iter().enumerate() {
            if self.is_constant_pattern(pat) {
                if current_seq_start.is_none() {
                    current_seq_start = Some(idx as u64);
                }
            } else {
                apply(&mut current_seq_start, idx as u64);
            }
        }

        apply(&mut current_seq_start, pattern.len() as u64);
        entries
    }

    fn should_optimize_subslice(&self, subslice: &[Box<Pat<'tcx>>]) -> bool {
        subslice.len() > 1 && subslice.iter().all(|p| self.is_constant_pattern(p))
    }

    fn is_constant_pattern(&self, pat: &Pat<'tcx>) -> bool {
        if let PatKind::Constant { value } = pat.kind
            && let Const::Ty(_, const_) = value
            && let ty::ConstKind::Value(_, valtree) = const_.kind()
            && let ty::ValTree::Leaf(_) = valtree
        {
            true
        } else {
            false
        }
    }

    fn extract_leaf(&self, pat: &Pat<'tcx>) -> ty::ValTree<'tcx> {
        if let PatKind::Constant { value } = pat.kind
            && let Const::Ty(_, const_) = value
            && let ty::ConstKind::Value(_, valtree) = const_.kind()
            && matches!(valtree, ty::ValTree::Leaf(_))
        {
            valtree
        } else {
            unreachable!()
        }
    }

    fn simplify_const_pattern_slice_into_valtree(
        &self,
        subslice: &[Box<Pat<'tcx>>],
    ) -> ty::ValTree<'tcx> {
        let leaves = subslice.iter().map(|p| self.extract_leaf(p));
        let interned = self.tcx.arena.alloc_from_iter(leaves);
        ty::ValTree::Branch(interned)
    }

    fn valtree_to_match_pair<'pat>(
        &mut self,
        source_pattern: &'pat Pat<'tcx>,
        valtree: ty::ValTree<'tcx>,
        place: PlaceBuilder<'tcx>,
        elem_ty: Ty<'tcx>,
        range: Range,
        do_own_slice: bool,
    ) -> MatchPairTree<'pat, 'tcx> {
        let tcx = self.tcx;
        let const_ty =
            Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, Ty::new_array(tcx, elem_ty, range.len()));

        let pat_ty = if do_own_slice { Ty::new_slice(tcx, elem_ty) } else { source_pattern.ty };
        let ty_const = ty::Const::new(tcx, ty::ConstKind::Value(const_ty, valtree));
        let value = Const::Ty(const_ty, ty_const);
        let test_case = TestCase::Constant { value, range: do_own_slice.then_some(range) };
        let pattern = tcx.arena.alloc(Pat {
            ty: pat_ty,
            span: source_pattern.span,
            kind: PatKind::Constant { value },
        });

        MatchPairTree {
            place: Some(place.to_place(self)),
            test_case,
            subpairs: Vec::new(),
            pattern,
        }
    }
}

impl<'pat, 'tcx> MatchPairTree<'pat, 'tcx> {
    /// Recursively builds a match pair tree for the given pattern and its
    /// subpatterns.
    pub(in crate::builder) fn for_pattern(
        mut place_builder: PlaceBuilder<'tcx>,
        pattern: &'pat Pat<'tcx>,
        cx: &mut Builder<'_, 'tcx>,
    ) -> MatchPairTree<'pat, 'tcx> {
        // Force the place type to the pattern's type.
        // FIXME(oli-obk): can we use this to simplify slice/array pattern hacks?
        if let Some(resolved) = place_builder.resolve_upvar(cx) {
            place_builder = resolved;
        }

        // Only add the OpaqueCast projection if the given place is an opaque type and the
        // expected type from the pattern is not.
        let may_need_cast = match place_builder.base() {
            PlaceBase::Local(local) => {
                let ty =
                    Place::ty_from(local, place_builder.projection(), &cx.local_decls, cx.tcx).ty;
                ty != pattern.ty && ty.has_opaque_types()
            }
            _ => true,
        };
        if may_need_cast {
            place_builder = place_builder.project(ProjectionElem::OpaqueCast(pattern.ty));
        }

        let place = place_builder.try_to_place(cx);
        let default_irrefutable = || TestCase::Irrefutable { binding: None, ascription: None };
        let mut subpairs = Vec::new();
        let test_case = match pattern.kind {
            PatKind::Wild | PatKind::Error(_) => default_irrefutable(),

            PatKind::Or { ref pats } => TestCase::Or {
                pats: pats.iter().map(|pat| FlatPat::new(place_builder.clone(), pat, cx)).collect(),
            },

            PatKind::Range(ref range) => {
                if range.is_full_range(cx.tcx) == Some(true) {
                    default_irrefutable()
                } else {
                    TestCase::Range(range)
                }
            }

            PatKind::Constant { value } => TestCase::Constant { value, range: None },

            PatKind::AscribeUserType {
                ascription: thir::Ascription { ref annotation, variance },
                ref subpattern,
                ..
            } => {
                // Apply the type ascription to the value at `match_pair.place`
                let ascription = place.map(|source| super::Ascription {
                    annotation: annotation.clone(),
                    source,
                    variance,
                });

                subpairs.push(MatchPairTree::for_pattern(place_builder, subpattern, cx));
                TestCase::Irrefutable { ascription, binding: None }
            }

            PatKind::Binding { mode, var, ref subpattern, .. } => {
                let binding = place.map(|source| super::Binding {
                    span: pattern.span,
                    source,
                    var_id: var,
                    binding_mode: mode,
                });

                if let Some(subpattern) = subpattern.as_ref() {
                    // this is the `x @ P` case; have to keep matching against `P` now
                    subpairs.push(MatchPairTree::for_pattern(place_builder, subpattern, cx));
                }
                TestCase::Irrefutable { ascription: None, binding }
            }

            PatKind::ExpandedConstant { subpattern: ref pattern, def_id: _, is_inline: false } => {
                subpairs.push(MatchPairTree::for_pattern(place_builder, pattern, cx));
                default_irrefutable()
            }
            PatKind::ExpandedConstant { subpattern: ref pattern, def_id, is_inline: true } => {
                // Apply a type ascription for the inline constant to the value at `match_pair.place`
                let ascription = place.map(|source| {
                    let span = pattern.span;
                    let parent_id = cx.tcx.typeck_root_def_id(cx.def_id.to_def_id());
                    let args = ty::InlineConstArgs::new(cx.tcx, ty::InlineConstArgsParts {
                        parent_args: ty::GenericArgs::identity_for_item(cx.tcx, parent_id),
                        ty: cx.infcx.next_ty_var(span),
                    })
                    .args;
                    let user_ty = cx.infcx.canonicalize_user_type_annotation(ty::UserType::new(
                        ty::UserTypeKind::TypeOf(def_id, ty::UserArgs { args, user_self_ty: None }),
                    ));
                    let annotation = ty::CanonicalUserTypeAnnotation {
                        inferred_ty: pattern.ty,
                        span,
                        user_ty: Box::new(user_ty),
                    };
                    super::Ascription { annotation, source, variance: ty::Contravariant }
                });

                subpairs.push(MatchPairTree::for_pattern(place_builder, pattern, cx));
                TestCase::Irrefutable { ascription, binding: None }
            }

            PatKind::Array { ref prefix, ref slice, ref suffix } => {
                cx.prefix_slice_suffix(
                    pattern,
                    &mut subpairs,
                    &place_builder,
                    prefix,
                    slice,
                    suffix,
                );
                default_irrefutable()
            }
            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                cx.prefix_slice_suffix(
                    pattern,
                    &mut subpairs,
                    &place_builder,
                    prefix,
                    slice,
                    suffix,
                );

                if prefix.is_empty() && slice.is_some() && suffix.is_empty() {
                    default_irrefutable()
                } else {
                    TestCase::Slice {
                        len: prefix.len() + suffix.len(),
                        variable_length: slice.is_some(),
                    }
                }
            }

            PatKind::Variant { adt_def, variant_index, args, ref subpatterns } => {
                let downcast_place = place_builder.downcast(adt_def, variant_index); // `(x as Variant)`
                subpairs = cx.field_match_pairs(downcast_place, subpatterns);

                let irrefutable = adt_def.variants().iter_enumerated().all(|(i, v)| {
                    i == variant_index
                        || !v
                            .inhabited_predicate(cx.tcx, adt_def)
                            .instantiate(cx.tcx, args)
                            .apply_ignore_module(cx.tcx, cx.infcx.typing_env(cx.param_env))
                }) && (adt_def.did().is_local()
                    || !adt_def.is_variant_list_non_exhaustive());
                if irrefutable {
                    default_irrefutable()
                } else {
                    TestCase::Variant { adt_def, variant_index }
                }
            }

            PatKind::Leaf { ref subpatterns } => {
                subpairs = cx.field_match_pairs(place_builder, subpatterns);
                default_irrefutable()
            }

            PatKind::Deref { ref subpattern } => {
                subpairs.push(MatchPairTree::for_pattern(place_builder.deref(), subpattern, cx));
                default_irrefutable()
            }

            PatKind::DerefPattern { ref subpattern, mutability } => {
                // Create a new temporary for each deref pattern.
                // FIXME(deref_patterns): dedup temporaries to avoid multiple `deref()` calls?
                let temp = cx.temp(
                    Ty::new_ref(cx.tcx, cx.tcx.lifetimes.re_erased, subpattern.ty, mutability),
                    pattern.span,
                );
                subpairs.push(MatchPairTree::for_pattern(
                    PlaceBuilder::from(temp).deref(),
                    subpattern,
                    cx,
                ));
                TestCase::Deref { temp, mutability }
            }

            PatKind::Never => TestCase::Never,
        };

        MatchPairTree { place, test_case, subpairs, pattern }
    }
}
