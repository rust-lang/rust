use std::ops;

use either::Either;
use rustc_middle::bug;
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
        top_pattern: &'pat Pat<'tcx>,
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

        if !prefix.is_empty() {
            let bounds = Range::from_start(0..prefix.len() as u64);
            let subpattern = bounds.apply(prefix);
            self.build_slice_branch(bounds, place, top_pattern, subpattern, min_length)
                .for_each(|pair| match_pairs.push(pair));
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

        if !suffix.is_empty() {
            let bounds = Range::from_end(0..suffix.len() as u64);
            let subpattern = bounds.apply(suffix);
            self.build_slice_branch(bounds, place, top_pattern, subpattern, min_length)
                .for_each(|pair| match_pairs.push(pair));
        }
    }

    // Traverses either side of a slice pattern (prefix/suffix) and yields an iterator of `MatchPairTree`s
    // to cover all it's constant and non-constant subpatterns.
    fn build_slice_branch<'pat, 'b>(
        &'b mut self,
        bounds: Range,
        place: &'b PlaceBuilder<'tcx>,
        top_pattern: &'pat Pat<'tcx>,
        pattern: &'pat [Box<Pat<'tcx>>],
        min_length: u64,
    ) -> impl Iterator<Item = MatchPairTree<'pat, 'tcx>> + use<'a, 'tcx, 'pat, 'b> {
        let entries = self.find_const_groups(pattern);

        entries.into_iter().map(move |entry| {
            // Common case handler for both non-constant and constant subpatterns not in a range.
            let mut build_single = |idx| {
                let subpattern = &pattern[idx as usize];
                let place = place.clone_project(ProjectionElem::ConstantIndex {
                    offset: bounds.shift_idx(idx),
                    min_length: pattern.len() as u64,
                    from_end: bounds.from_end,
                });

                MatchPairTree::for_pattern(place, subpattern, self)
            };

            match entry {
                Either::Right(range) if range.end - range.start > 1 => {
                    // Figure out which subslice of our already sliced pattern we're looking at.
                    let subpattern = &pattern[range.start as usize..range.end as usize];
                    let elem_ty = subpattern[0].ty;

                    // Right, we 've found a group of constant patterns worth grouping for later.
                    // We'll collect all the leaves we can find and create a single `ValTree` out of them.
                    let valtree = self.simplify_const_pattern_slice_into_valtree(subpattern);
                    self.valtree_to_match_pair(
                        top_pattern,
                        valtree,
                        place.clone(),
                        elem_ty,
                        bounds.shift_range(range),
                        min_length,
                    )
                }
                Either::Right(range) => build_single(range.start),
                Either::Left(idx) => build_single(idx),
            }
        })
    }

    // Given a partial view of the elements in a slice pattern, returns a list
    // with left denoting non-constant element indices and right denoting ranges of constant elements.
    fn find_const_groups(&self, pattern: &[Box<Pat<'tcx>>]) -> Vec<Either<u64, ops::Range<u64>>> {
        let mut entries = Vec::new();
        let mut current_seq_start = None;

        for (idx, pat) in pattern.iter().enumerate() {
            if self.is_constant_pattern(pat) {
                if current_seq_start.is_none() {
                    current_seq_start = Some(idx as u64);
                } else {
                    continue;
                }
            } else {
                if let Some(start) = current_seq_start {
                    entries.push(Either::Right(start..idx as u64));
                    current_seq_start = None;
                }
                entries.push(Either::Left(idx as u64));
            }
        }

        if let Some(start) = current_seq_start {
            entries.push(Either::Right(start..pattern.len() as u64));
        }

        entries
    }

    // Checks if a pattern is constant and represented by a single scalar leaf.
    fn is_constant_pattern(&self, pat: &Pat<'tcx>) -> bool {
        if let PatKind::Constant { value } = pat.kind
            && let Const::Ty(_, const_) = value
            && let ty::ConstKind::Value(cv) = const_.kind()
            && let ty::ValTree::Leaf(_) = cv.valtree
        {
            true
        } else {
            false
        }
    }

    // Extract the `ValTree` from a constant pattern.
    // You must ensure that the pattern is a constant pattern before calling this function or it will panic.
    fn extract_leaf(&self, pat: &Pat<'tcx>) -> ty::ValTree<'tcx> {
        if let PatKind::Constant { value } = pat.kind
            && let Const::Ty(_, const_) = value
            && let ty::ConstKind::Value(cv) = const_.kind()
            && matches!(cv.valtree, ty::ValTree::Leaf(_))
        {
            cv.valtree
        } else {
            bug!("expected constant pattern, got {:?}", pat)
        }
    }

    // Simplifies a slice of constant patterns into a single flattened `ValTree`.
    fn simplify_const_pattern_slice_into_valtree(
        &self,
        subslice: &[Box<Pat<'tcx>>],
    ) -> ty::ValTree<'tcx> {
        let leaves = subslice.iter().map(|p| self.extract_leaf(p));
        let interned = self.tcx.arena.alloc_from_iter(leaves);
        ty::ValTree::Branch(interned)
    }

    // Given a `ValTree` representing a slice of constant patterns, returns a `MatchPairTree`
    // representing the slice pattern, providing as much info about subsequences in the slice as possible
    // to later lowering stages.
    fn valtree_to_match_pair<'pat>(
        &mut self,
        source_pattern: &'pat Pat<'tcx>,
        valtree: ty::ValTree<'tcx>,
        place: PlaceBuilder<'tcx>,
        elem_ty: Ty<'tcx>,
        range: Range,
        min_length: u64,
    ) -> MatchPairTree<'pat, 'tcx> {
        let tcx = self.tcx;
        let leaves = match valtree {
            ty::ValTree::Leaf(_) => bug!("expected branch, got leaf"),
            ty::ValTree::Branch(leaves) => leaves,
        };

        assert!(range.len() == leaves.len() as u64);
        let mut subpairs = Vec::new();
        let mut were_merged = 0;

        if elem_ty == tcx.types.u8 {
            let leaf_bits = |leaf: ty::ValTree<'tcx>| match leaf {
                ty::ValTree::Leaf(scalar) => scalar.to_u8(),
                _ => bug!("found unflatted valtree"),
            };

            let mut fuse_group = |first_idx, len| {
                were_merged += len;

                let data = leaves[first_idx..first_idx + len]
                    .iter()
                    .rev()
                    .copied()
                    .map(leaf_bits)
                    .fold(0u32, |acc, x| (acc << 8) | u32::from(x));

                let fused_ty = match len {
                    2 => tcx.types.u16,
                    3 | 4 => tcx.types.u32,
                    _ => unreachable!(),
                };

                let scalar = match len {
                    2 => ty::ScalarInt::from(data as u16),
                    3 | 4 => ty::ScalarInt::from(data),
                    _ => unreachable!(),
                };

                let valtree = ty::ValTree::Leaf(scalar);
                let ty_const =
                    ty::Const::new(tcx, ty::ConstKind::Value(ty::Value { ty: fused_ty, valtree }));

                let value = Const::Ty(fused_ty, ty_const);
                let test_case = TestCase::FusedConstant { value, fused: len as u64 };

                let pattern = tcx.arena.alloc(Pat {
                    ty: fused_ty,
                    span: source_pattern.span,
                    kind: PatKind::Constant { value },
                });

                let place = place
                    .clone_project(ProjectionElem::ConstantIndex {
                        offset: range.shift_idx(first_idx as u64),
                        min_length,
                        from_end: range.from_end,
                    })
                    .to_place(self);

                subpairs.push(MatchPairTree {
                    place: Some(place),
                    test_case,
                    subpairs: Vec::new(),
                    pattern,
                });
            };

            let indices = |group_size, skip| {
                (skip..usize::MAX)
                    .take_while(move |i| i * group_size + (group_size - 1) < leaves.len())
            };

            let mut skip = 0;
            for i in (2..=4).rev() {
                for idx in indices(i, skip) {
                    fuse_group(idx * i, i);
                    skip += i;
                }
            }
        }

        for (idx, leaf) in leaves.iter().enumerate().skip(were_merged) {
            let ty_const = ty::Const::new(
                tcx,
                ty::ConstKind::Value(ty::Value { ty: elem_ty, valtree: *leaf }),
            );
            let value = Const::Ty(elem_ty, ty_const);
            let test_case = TestCase::Constant { value };

            let pattern = tcx.arena.alloc(Pat {
                ty: elem_ty,
                span: source_pattern.span,
                kind: PatKind::Constant { value },
            });

            let place = place
                .clone_project(ProjectionElem::ConstantIndex {
                    offset: range.start + idx as u64,
                    min_length,
                    from_end: range.from_end,
                })
                .to_place(self);

            subpairs.push(MatchPairTree {
                place: Some(place),
                test_case,
                subpairs: Vec::new(),
                pattern,
            });
        }

        MatchPairTree {
            place: None,
            test_case: TestCase::Irrefutable { binding: None, ascription: None },
            subpairs,
            pattern: source_pattern,
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

            PatKind::Constant { value } => TestCase::Constant { value },

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
