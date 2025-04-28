use std::sync::Arc;

use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};

use crate::builder::Builder;
use crate::builder::expr::as_place::{PlaceBase, PlaceBuilder};
use crate::builder::matches::{FlatPat, MatchPairTree, PatternExtraData, TestCase};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Builds and pushes [`MatchPairTree`] subtrees, one for each pattern in
    /// `subpatterns`, representing the fields of a [`PatKind::Variant`] or
    /// [`PatKind::Leaf`].
    ///
    /// Used internally by [`MatchPairTree::for_pattern`].
    fn field_match_pairs(
        &mut self,
        match_pairs: &mut Vec<MatchPairTree<'tcx>>,
        extra_data: &mut PatternExtraData<'tcx>,
        place: PlaceBuilder<'tcx>,
        subpatterns: &[FieldPat<'tcx>],
    ) {
        for fieldpat in subpatterns {
            let place = place.clone_project(PlaceElem::Field(fieldpat.field, fieldpat.pattern.ty));
            MatchPairTree::for_pattern(place, &fieldpat.pattern, self, match_pairs, extra_data);
        }
    }

    /// Builds [`MatchPairTree`] subtrees for the prefix/middle/suffix parts of an
    /// array pattern or slice pattern, and adds those trees to `match_pairs`.
    ///
    /// Used internally by [`MatchPairTree::for_pattern`].
    fn prefix_slice_suffix(
        &mut self,
        match_pairs: &mut Vec<MatchPairTree<'tcx>>,
        extra_data: &mut PatternExtraData<'tcx>,
        place: &PlaceBuilder<'tcx>,
        prefix: &[Pat<'tcx>],
        opt_slice: &Option<Box<Pat<'tcx>>>,
        suffix: &[Pat<'tcx>],
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

        for (idx, subpattern) in prefix.iter().enumerate() {
            let elem =
                ProjectionElem::ConstantIndex { offset: idx as u64, min_length, from_end: false };
            let place = place.clone_project(elem);
            MatchPairTree::for_pattern(place, subpattern, self, match_pairs, extra_data)
        }

        if let Some(subslice_pat) = opt_slice {
            let suffix_len = suffix.len() as u64;
            let subslice = place.clone_project(PlaceElem::Subslice {
                from: prefix.len() as u64,
                to: if exact_size { min_length - suffix_len } else { suffix_len },
                from_end: !exact_size,
            });
            MatchPairTree::for_pattern(subslice, subslice_pat, self, match_pairs, extra_data);
        }

        for (idx, subpattern) in suffix.iter().rev().enumerate() {
            let end_offset = (idx + 1) as u64;
            let elem = ProjectionElem::ConstantIndex {
                offset: if exact_size { min_length - end_offset } else { end_offset },
                min_length,
                from_end: !exact_size,
            };
            let place = place.clone_project(elem);
            MatchPairTree::for_pattern(place, subpattern, self, match_pairs, extra_data)
        }
    }
}

impl<'tcx> MatchPairTree<'tcx> {
    /// Recursively builds a match pair tree for the given pattern and its
    /// subpatterns.
    pub(super) fn for_pattern(
        mut place_builder: PlaceBuilder<'tcx>,
        pattern: &Pat<'tcx>,
        cx: &mut Builder<'_, 'tcx>,
        match_pairs: &mut Vec<Self>, // Newly-created nodes are added to this vector
        extra_data: &mut PatternExtraData<'tcx>, // Bindings/ascriptions are added here
    ) {
        // Force the place type to the pattern's type.
        // FIXME(oli-obk): can we use this to simplify slice/array pattern hacks?
        if let Some(resolved) = place_builder.resolve_upvar(cx) {
            place_builder = resolved;
        }

        if !cx.tcx.next_trait_solver_globally() {
            // Only add the OpaqueCast projection if the given place is an opaque type and the
            // expected type from the pattern is not.
            let may_need_cast = match place_builder.base() {
                PlaceBase::Local(local) => {
                    let ty =
                        Place::ty_from(local, place_builder.projection(), &cx.local_decls, cx.tcx)
                            .ty;
                    ty != pattern.ty && ty.has_opaque_types()
                }
                _ => true,
            };
            if may_need_cast {
                place_builder = place_builder.project(ProjectionElem::OpaqueCast(pattern.ty));
            }
        }

        let place = place_builder.try_to_place(cx);
        let mut subpairs = Vec::new();
        let test_case = match pattern.kind {
            PatKind::Missing | PatKind::Wild | PatKind::Error(_) => None,

            PatKind::Or { ref pats } => Some(TestCase::Or {
                pats: pats.iter().map(|pat| FlatPat::new(place_builder.clone(), pat, cx)).collect(),
            }),

            PatKind::Range(ref range) => {
                if range.is_full_range(cx.tcx) == Some(true) {
                    None
                } else {
                    Some(TestCase::Range(Arc::clone(range)))
                }
            }

            PatKind::Constant { value } => Some(TestCase::Constant { value }),

            PatKind::AscribeUserType {
                ascription: Ascription { ref annotation, variance },
                ref subpattern,
                ..
            } => {
                MatchPairTree::for_pattern(
                    place_builder,
                    subpattern,
                    cx,
                    &mut subpairs,
                    extra_data,
                );

                // Apply the type ascription to the value at `match_pair.place`
                if let Some(source) = place {
                    let annotation = annotation.clone();
                    extra_data.ascriptions.push(super::Ascription { source, annotation, variance });
                }

                None
            }

            PatKind::Binding { mode, var, ref subpattern, .. } => {
                // In order to please the borrow checker, when lowering a pattern
                // like `x @ subpat` we must establish any bindings in `subpat`
                // before establishing the binding for `x`.
                //
                // For example (from #69971):
                //
                // ```ignore (illustrative)
                // struct NonCopyStruct {
                //     copy_field: u32,
                // }
                //
                // fn foo1(x: NonCopyStruct) {
                //     let y @ NonCopyStruct { copy_field: z } = x;
                //     // the above should turn into
                //     let z = x.copy_field;
                //     let y = x;
                // }
                // ```

                // First, recurse into the subpattern, if any.
                if let Some(subpattern) = subpattern.as_ref() {
                    // this is the `x @ P` case; have to keep matching against `P` now
                    MatchPairTree::for_pattern(
                        place_builder,
                        subpattern,
                        cx,
                        &mut subpairs,
                        extra_data,
                    );
                }

                // Then push this binding, after any bindings in the subpattern.
                if let Some(source) = place {
                    extra_data.bindings.push(super::Binding {
                        span: pattern.span,
                        source,
                        var_id: var,
                        binding_mode: mode,
                    });
                }

                None
            }

            PatKind::ExpandedConstant { subpattern: ref pattern, .. } => {
                MatchPairTree::for_pattern(place_builder, pattern, cx, &mut subpairs, extra_data);
                None
            }

            PatKind::Array { ref prefix, ref slice, ref suffix } => {
                cx.prefix_slice_suffix(
                    &mut subpairs,
                    extra_data,
                    &place_builder,
                    prefix,
                    slice,
                    suffix,
                );
                None
            }
            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                cx.prefix_slice_suffix(
                    &mut subpairs,
                    extra_data,
                    &place_builder,
                    prefix,
                    slice,
                    suffix,
                );

                if prefix.is_empty() && slice.is_some() && suffix.is_empty() {
                    None
                } else {
                    Some(TestCase::Slice {
                        len: prefix.len() + suffix.len(),
                        variable_length: slice.is_some(),
                    })
                }
            }

            PatKind::Variant { adt_def, variant_index, args, ref subpatterns } => {
                let downcast_place = place_builder.downcast(adt_def, variant_index); // `(x as Variant)`
                cx.field_match_pairs(&mut subpairs, extra_data, downcast_place, subpatterns);

                let irrefutable = adt_def.variants().iter_enumerated().all(|(i, v)| {
                    i == variant_index
                        || !v.inhabited_predicate(cx.tcx, adt_def).instantiate(cx.tcx, args).apply(
                            cx.tcx,
                            cx.infcx.typing_env(cx.param_env),
                            cx.def_id.into(),
                        )
                }) && !adt_def.variant_list_has_applicable_non_exhaustive();
                if irrefutable { None } else { Some(TestCase::Variant { adt_def, variant_index }) }
            }

            PatKind::Leaf { ref subpatterns } => {
                cx.field_match_pairs(&mut subpairs, extra_data, place_builder, subpatterns);
                None
            }

            PatKind::Deref { ref subpattern } => {
                MatchPairTree::for_pattern(
                    place_builder.deref(),
                    subpattern,
                    cx,
                    &mut subpairs,
                    extra_data,
                );
                None
            }

            PatKind::DerefPattern { ref subpattern, mutability } => {
                // Create a new temporary for each deref pattern.
                // FIXME(deref_patterns): dedup temporaries to avoid multiple `deref()` calls?
                let temp = cx.temp(
                    Ty::new_ref(cx.tcx, cx.tcx.lifetimes.re_erased, subpattern.ty, mutability),
                    pattern.span,
                );
                MatchPairTree::for_pattern(
                    PlaceBuilder::from(temp).deref(),
                    subpattern,
                    cx,
                    &mut subpairs,
                    extra_data,
                );
                Some(TestCase::Deref { temp, mutability })
            }

            PatKind::Never => Some(TestCase::Never),
        };

        if let Some(test_case) = test_case {
            // This pattern is refutable, so push a new match-pair node.
            match_pairs.push(MatchPairTree {
                place,
                test_case,
                subpairs,
                pattern_ty: pattern.ty,
                pattern_span: pattern.span,
            })
        } else {
            // This pattern is irrefutable, so it doesn't need its own match-pair node.
            // Just push its refutable subpatterns instead, if any.
            match_pairs.extend(subpairs);
        }
    }
}
