use std::sync::Arc;

use rustc_abi::FieldIdx;
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};

use crate::builder::Builder;
use crate::builder::expr::as_place::{PlaceBase, PlaceBuilder};
use crate::builder::matches::{
    FlatPat, MatchPairTree, PatConstKind, PatternExtraData, SliceLenOp, TestableCase,
};

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
            let place_ty = place_resolved.ty(&self.local_decls, tcx).ty;
            match place_ty.kind() {
                ty::Array(_, length) => {
                    if let Some(length) = length.try_to_target_usize(tcx) {
                        (length, true)
                    } else {
                        // This can happen when the array length is a generic const
                        // expression that couldn't be evaluated (e.g., due to an error).
                        // Since there's already a compilation error, we use a fallback
                        // to avoid an ICE.
                        tcx.dcx().span_delayed_bug(
                            tcx.def_span(self.def_id),
                            "array length in pattern couldn't be evaluated",
                        );
                        ((prefix.len() + suffix.len()).try_into().unwrap(), false)
                    }
                }
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

        // Apply any type ascriptions to the value at `match_pair.place`.
        if let Some(place) = place
            && let Some(extra) = &pattern.extra
        {
            for &Ascription { ref annotation, variance } in &extra.ascriptions {
                extra_data.ascriptions.push(super::Ascription {
                    source: place,
                    annotation: annotation.clone(),
                    variance,
                });
            }
        }

        let mut subpairs = Vec::new();
        let testable_case = match pattern.kind {
            PatKind::Missing | PatKind::Wild | PatKind::Error(_) => None,

            PatKind::Or { ref pats } => {
                let pats: Box<[FlatPat<'tcx>]> =
                    pats.iter().map(|pat| FlatPat::new(place_builder.clone(), pat, cx)).collect();
                if !pats[0].extra_data.bindings.is_empty() {
                    // Hold a place for any bindings established in (possibly-nested) or-patterns.
                    // By only holding a place when bindings are present, we skip over any
                    // or-patterns that will be simplified by `merge_trivial_subcandidates`. In
                    // other words, we can assume this expands into subcandidates.
                    // FIXME(@dianne): this needs updating/removing if we always merge or-patterns
                    extra_data.bindings.push(super::SubpatternBindings::FromOrPattern);
                }
                Some(TestableCase::Or { pats })
            }

            PatKind::Range(ref range) => {
                if range.is_full_range(cx.tcx) == Some(true) {
                    None
                } else {
                    Some(TestableCase::Range(Arc::clone(range)))
                }
            }

            PatKind::Constant { value } => {
                // CAUTION: The type of the pattern node (`pattern.ty`) is
                // _often_ the same as the type of the const value (`value.ty`),
                // but there are some cases where those types differ
                // (e.g. when `deref!(..)` patterns interact with `String`).

                // Classify the constant-pattern into further kinds, to
                // reduce the number of ad-hoc type tests needed later on.
                let pat_ty = pattern.ty;
                let const_kind = if pat_ty.is_bool() {
                    PatConstKind::Bool
                } else if pat_ty.is_integral() || pat_ty.is_char() {
                    PatConstKind::IntOrChar
                } else if pat_ty.is_floating_point() {
                    PatConstKind::Float
                } else if pat_ty.is_str() {
                    // Deref-patterns can cause string-literal patterns to have
                    // type `str` instead of the usual `&str`.
                    if !cx.tcx.features().deref_patterns() {
                        span_bug!(
                            pattern.span,
                            "const pattern has type `str` but deref_patterns is not enabled"
                        );
                    }
                    PatConstKind::String
                } else if pat_ty.is_imm_ref_str() {
                    PatConstKind::String
                } else {
                    // FIXME(Zalathar): This still covers several different
                    // categories (e.g. raw pointer, pattern-type)
                    // which could be split out into their own kinds.
                    PatConstKind::Other
                };
                Some(TestableCase::Constant { value, kind: const_kind })
            }

            PatKind::Binding { mode, var, is_shorthand, ref subpattern, .. } => {
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
                    extra_data.bindings.push(super::SubpatternBindings::One(super::Binding {
                        span: pattern.span,
                        source,
                        var_id: var,
                        binding_mode: mode,
                        is_shorthand,
                    }));
                }

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
                    // This pattern is shaped like `[..]`. It can match a slice
                    // of any length, so no length test is needed.
                    None
                } else {
                    // Any other shape of slice pattern requires a length test.
                    // Slice patterns with a `..` subpattern require a minimum
                    // length; those without `..` require an exact length.
                    Some(TestableCase::Slice {
                        len: u64::try_from(prefix.len() + suffix.len()).unwrap(),
                        op: if slice.is_some() {
                            SliceLenOp::GreaterOrEqual
                        } else {
                            SliceLenOp::Equal
                        },
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
                if irrefutable {
                    None
                } else {
                    Some(TestableCase::Variant { adt_def, variant_index })
                }
            }

            PatKind::Leaf { ref subpatterns } => {
                cx.field_match_pairs(&mut subpairs, extra_data, place_builder, subpatterns);
                None
            }

            // FIXME: Pin-patterns should probably have their own pattern kind,
            // instead of overloading `PatKind::Deref` via the pattern type.
            PatKind::Deref { ref subpattern }
                if let Some(ref_ty) = pattern.ty.pinned_ty()
                    && ref_ty.is_ref() =>
            {
                MatchPairTree::for_pattern(
                    place_builder.field(FieldIdx::ZERO, ref_ty).deref(),
                    subpattern,
                    cx,
                    &mut subpairs,
                    extra_data,
                );
                None
            }

            PatKind::Deref { ref subpattern }
            | PatKind::DerefPattern { ref subpattern, borrow: DerefPatBorrowMode::Box } => {
                MatchPairTree::for_pattern(
                    place_builder.deref(),
                    subpattern,
                    cx,
                    &mut subpairs,
                    extra_data,
                );
                None
            }

            PatKind::DerefPattern {
                ref subpattern,
                borrow: DerefPatBorrowMode::Borrow(mutability),
            } => {
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
                Some(TestableCase::Deref { temp, mutability })
            }

            PatKind::Never => Some(TestableCase::Never),
        };

        if let Some(testable_case) = testable_case {
            // This pattern is refutable, so push a new match-pair node.
            //
            // Note: unless test_case is TestCase::Or, place must not be None.
            // This means that the closure capture analysis in
            // rustc_hir_typeck::upvar, and in particular the pattern handling
            // code of ExprUseVisitor, must capture all of the places we'll use.
            // Make sure to keep these two parts in sync!
            match_pairs.push(MatchPairTree {
                place,
                testable_case,
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
