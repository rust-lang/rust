use std::sync::Arc;

use rustc_abi::FieldIdx;
use rustc_middle::mir::{Pinnedness, Place, PlaceElem, ProjectionElem};
use rustc_middle::span_bug;
use rustc_middle::thir::{Ascription, DerefPatBorrowMode, FieldPat, Pat, PatKind};
use rustc_middle::ty::{self, Ty, TypeVisitableExt};
use rustc_span::Span;

use crate::builder::Builder;
use crate::builder::expr::as_place::{PlaceBase, PlaceBuilder};
use crate::builder::matches::{
    FlatPat, MatchPairKind, MatchPairTree, PatConstKind, PatternExtraData, SliceLenOp, TestableCase,
};

/// For an array or slice pattern's subpatterns (prefix/slice/suffix), returns a list
/// of those subpatterns, each paired with a suitably-projected [`PlaceBuilder`].
fn prefix_slice_suffix<'a, 'tcx>(
    place: &PlaceBuilder<'tcx>,
    array_len: Option<u64>, // Some for array patterns; None for slice patterns
    prefix: &'a [Pat<'tcx>],
    opt_slice: &'a Option<Box<Pat<'tcx>>>,
    suffix: &'a [Pat<'tcx>],
) -> Vec<(PlaceBuilder<'tcx>, &'a Pat<'tcx>)> {
    let prefix_len = u64::try_from(prefix.len()).unwrap();
    let suffix_len = u64::try_from(suffix.len()).unwrap();

    let mut output_pairs =
        Vec::with_capacity(prefix.len() + usize::from(opt_slice.is_some()) + suffix.len());

    // For slice patterns with a `..` followed by 0 or more suffix subpatterns,
    // the actual slice index of those subpatterns isn't statically known, so
    // we have to index them relative to the end of the slice.
    //
    // For array patterns, all subpatterns are indexed relative to the start.
    let (min_length, is_array) = match array_len {
        Some(len) => (len, true),
        None => (prefix_len + suffix_len, false),
    };

    for (offset, prefix_subpat) in (0u64..).zip(prefix) {
        let elem = ProjectionElem::ConstantIndex { offset, min_length, from_end: false };
        let subplace = place.clone_project(elem);
        output_pairs.push((subplace, prefix_subpat));
    }

    if let Some(slice_subpat) = opt_slice {
        let elem = PlaceElem::Subslice {
            from: prefix_len,
            to: if is_array { min_length - suffix_len } else { suffix_len },
            from_end: !is_array,
        };
        let subplace = place.clone_project(elem);
        output_pairs.push((subplace, slice_subpat));
    }

    for (offset_from_end, suffix_subpat) in (1u64..).zip(suffix.iter().rev()) {
        let elem = ProjectionElem::ConstantIndex {
            offset: if is_array { min_length - offset_from_end } else { offset_from_end },
            min_length,
            from_end: !is_array,
        };
        let subplace = place.clone_project(elem);
        output_pairs.push((subplace, suffix_subpat));
    }

    output_pairs
}

impl<'tcx> FlatPat<'tcx> {
    /// Creates a `FlatPat` containing a simplified [`MatchPairTree`] list/forest
    /// for the given pattern.
    pub(crate) fn new(
        place: PlaceBuilder<'tcx>,
        pattern: &Pat<'tcx>,
        cx: &mut Builder<'_, 'tcx>,
    ) -> Self {
        // Recursively lower the THIR pattern into an intermediate form,
        // then flatten into a `FlatPat`.
        let inter_pat = InterPat::lower_thir_pat(cx, place, pattern);
        FlatPat::from_inter_pat(inter_pat)
    }

    /// Squashes an [`InterPat`] into a [`FlatPat`].
    ///
    /// This is a separate function because it also needs to be called recursively
    /// when squashing any or-patterns.
    fn from_inter_pat(inter_pat: InterPat<'tcx>) -> Self {
        let mut match_pairs = vec![];
        let mut extra_data = PatternExtraData {
            span: inter_pat.pattern_span,
            bindings: vec![],
            ascriptions: vec![],
            is_never: inter_pat.is_never,
        };
        squash_inter_pat(inter_pat, &mut match_pairs, &mut extra_data);

        FlatPat { match_pairs, extra_data }
    }
}

/// Recursively squashes an [`InterPat`] into a forest of refutable [`MatchPairTree`]
/// nodes, while accumulating ascriptions and bindings.
fn squash_inter_pat<'tcx>(
    inter_pat: InterPat<'tcx>,
    match_pairs: &mut Vec<MatchPairTree<'tcx>>, // Newly-created nodes are added to this vector
    extra_data: &mut PatternExtraData<'tcx>,    // Bindings/ascriptions are added here
) {
    // Destructure exhaustively to make sure we don't miss any fields.
    // The `is_never` field is not needed by `MatchPairTree` forests.
    let InterPat { kind, ascriptions, pattern_span, is_never: _ } = inter_pat;

    // Type ascriptions can appear regardless of whether the node is an or-pattern.
    extra_data.ascriptions.extend(ascriptions);

    // Or patterns, refutable patterns, and irrefutable patterns all have different handling.
    match kind {
        InterPatKind::Or { or_subpats } => {
            let or_subpats = or_subpats
                .into_iter()
                .map(|subpat| FlatPat::from_inter_pat(subpat))
                .collect::<Box<[_]>>();

            if !or_subpats[0].extra_data.bindings.is_empty() {
                // Hold a place for any bindings established in (possibly-nested) or-patterns.
                // By only holding a place when bindings are present, we skip over any
                // or-patterns that will be simplified by `merge_trivial_subcandidates`. In
                // other words, we can assume this expands into subcandidates.
                // FIXME(@dianne): this needs updating/removing if we always merge or-patterns
                extra_data.bindings.push(super::SubpatternBindings::FromOrPattern);
            }

            match_pairs
                .push(MatchPairTree { kind: MatchPairKind::Or { or_subpats }, pattern_span });
        }

        InterPatKind::Refutable { place, testable_case, subpats } => {
            // Recursively squash any subpatterns into refutable `MatchPairTree` forests,
            // which will become the children of a new node.
            let mut subpairs = vec![];
            for subpat in subpats {
                squash_inter_pat(subpat, &mut subpairs, extra_data);
            }

            // This pattern is refutable, so push a new match-pair node.
            match_pairs.push(MatchPairTree {
                kind: MatchPairKind::Testable { place, testable_case, subpairs },
                pattern_span,
            });
        }

        InterPatKind::Irrefutable { subpats, binding } => {
            // Recursively squash any subpatterns into refutable `MatchPairTree` forests.
            // This must happen _before_ pushing the binding, as described by the binding step.
            for subpat in subpats {
                // For irrefutable nodes, squash directly into the caller's match pairs.
                squash_inter_pat(subpat, match_pairs, extra_data);
            }

            // If present, the binding must be pushed _after_ traversing subpatterns.
            // This is so that when lowering something like `x @ NonCopy { copy_field }`,
            // the binding to `copy_field` will occur before the binding for `x`.
            // See <https://github.com/rust-lang/rust/issues/69971> for more background.
            if let Some(binding) = binding {
                extra_data.bindings.push(super::SubpatternBindings::One(binding));
            }
        }
    }
}

/// "Intermediate pattern", a partly-lowered THIR [`Pat`] that has not yet been
/// squashed into a forest of refutable [`MatchPairTree`] nodes.
struct InterPat<'tcx> {
    kind: InterPatKind<'tcx>,

    ascriptions: Vec<super::Ascription<'tcx>>,
    /// Span field of the THIR pattern this node was created from.
    pattern_span: Span,
    /// True if this pattern can never match, because all of its alternatives
    /// contain a `!` pattern.
    is_never: bool,
}

enum InterPatKind<'tcx> {
    Or {
        /// The alternatives of an or-pattern, e.g. `P` and `Q` in `P | Q`.
        or_subpats: Box<[InterPat<'tcx>]>,
    },

    /// Pattern node that performs some kind of test on a place.
    Refutable {
        /// Place that this pattern node will test.
        place: Place<'tcx>,
        /// Testable condition to compare the place to (e.g. "is 3" or "is Some").
        testable_case: TestableCase<'tcx>,
        /// Immediate subpatterns.
        subpats: Vec<InterPat<'tcx>>,
    },

    /// Pattern node that doesn't test anything, though it might have refutable descendants.
    Irrefutable {
        /// Immediate subpatterns.
        subpats: Vec<InterPat<'tcx>>,
        /// Binding to establish for a [`PatKind::Binding`] node.
        binding: Option<super::Binding<'tcx>>,
    },
}

impl<'tcx> InterPat<'tcx> {
    fn lower_thir_pat(
        cx: &mut Builder<'_, 'tcx>,
        mut place_builder: PlaceBuilder<'tcx>,
        pattern: &Pat<'tcx>,
    ) -> Self {
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
        let mut ascriptions = vec![];
        if let Some(place) = place
            && let Some(extra) = &pattern.extra
        {
            ascriptions.extend(extra.ascriptions.iter().map(
                |&Ascription { ref annotation, variance }| super::Ascription {
                    source: place,
                    annotation: annotation.clone(),
                    variance,
                },
            ));
        }

        // For refutable nodes a place must be available, either because it is not a
        // closure upvar or because it was captured.
        let unwrap_place = || place.expect("refutable patterns must have captured a place");

        let kind: InterPatKind<'_> = match pattern.kind {
            PatKind::Missing | PatKind::Wild | PatKind::Error(_) => {
                InterPatKind::Irrefutable { subpats: vec![], binding: None }
            }

            PatKind::Or { ref pats } => {
                let or_subpats = pats
                    .iter()
                    .map(|subpat| InterPat::lower_thir_pat(cx, place_builder.clone(), subpat))
                    .collect::<Box<[_]>>();
                InterPatKind::Or { or_subpats }
            }

            PatKind::Range(ref range) => {
                assert_eq!(pattern.ty, range.ty);
                if range.is_full_range(cx.tcx) == Some(true) {
                    InterPatKind::Irrefutable { subpats: vec![], binding: None }
                } else {
                    InterPatKind::Refutable {
                        place: unwrap_place(),
                        testable_case: TestableCase::Range(Arc::clone(range)),
                        subpats: vec![],
                    }
                }
            }

            PatKind::Constant { value } => {
                assert_eq!(pattern.ty, value.ty);

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
                    PatConstKind::String
                } else {
                    // FIXME(Zalathar): This still covers several different
                    // categories (e.g. raw pointer, pattern-type)
                    // which could be split out into their own kinds.
                    PatConstKind::Other
                };

                InterPatKind::Refutable {
                    place: unwrap_place(),
                    testable_case: TestableCase::Constant { value, kind: const_kind },
                    subpats: vec![],
                }
            }

            PatKind::Binding { mode, var, is_shorthand, ref subpattern, .. } => {
                // First, recurse into the subpattern, if any.
                // This is the `x @ P` case; have to keep matching against `P` now.
                let subpat: Option<InterPat<'_>> = subpattern
                    .as_deref()
                    .map(|subpattern| InterPat::lower_thir_pat(cx, place_builder, subpattern));

                // Then push this binding, after any bindings in the subpattern.
                let binding = place.map(|place| super::Binding {
                    span: pattern.span,
                    source: place,
                    var_id: var,
                    binding_mode: mode,
                    is_shorthand,
                });
                InterPatKind::Irrefutable { subpats: Vec::from_iter(subpat), binding }
            }

            PatKind::Array { ref prefix, ref slice, ref suffix } => {
                // Determine the statically-known length of the array type being matched.
                // This should always succeed for legal programs, but could fail for
                // erroneous programs (e.g. the type is `[u8; const { panic!() }]`),
                // so take care not to ICE if this fails.
                let array_len = match pattern.ty.kind() {
                    ty::Array(_, len) => len.try_to_target_usize(cx.tcx),
                    _ => None,
                };

                let mut subpats = vec![];
                if let Some(array_len) = array_len {
                    for (subplace, subpat) in
                        prefix_slice_suffix(&place_builder, Some(array_len), prefix, slice, suffix)
                    {
                        subpats.push(InterPat::lower_thir_pat(cx, subplace, subpat));
                    }
                } else {
                    // If the array length couldn't be determined, ignore the
                    // subpatterns and delayed-assert that compilation will fail.
                    cx.tcx.dcx().span_delayed_bug(
                        pattern.span,
                        format!(
                            "array length in pattern couldn't be determined for ty={:?}",
                            pattern.ty
                        ),
                    );
                }

                InterPatKind::Irrefutable { subpats, binding: None }
            }
            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                let mut subpats = vec![];
                for (subplace, subpat) in
                    prefix_slice_suffix(&place_builder, None, prefix, slice, suffix)
                {
                    subpats.push(InterPat::lower_thir_pat(cx, subplace, subpat));
                }

                if prefix.is_empty() && slice.is_some() && suffix.is_empty() {
                    // A slice pattern shaped like `[..]` is irrefutable.
                    // It can match a slice of any length, so no length test is needed.
                    InterPatKind::Irrefutable { subpats, binding: None }
                } else {
                    // Any other shape of slice pattern requires a length test.
                    // Slice patterns with a `..` subpattern require a minimum
                    // length; those without `..` require an exact length.
                    let testable_case = TestableCase::Slice {
                        len: u64::try_from(prefix.len() + suffix.len()).unwrap(),
                        op: if slice.is_some() {
                            SliceLenOp::GreaterOrEqual
                        } else {
                            SliceLenOp::Equal
                        },
                    };
                    InterPatKind::Refutable { place: unwrap_place(), testable_case, subpats }
                }
            }

            PatKind::Variant { adt_def, variant_index, args: _, ref subpatterns } => {
                let downcast_place = place_builder.downcast(adt_def, variant_index); // `(x as Variant)`
                let mut subpats = vec![];
                for &FieldPat { field, pattern: ref subpat } in subpatterns {
                    let subplace = downcast_place.clone_project(PlaceElem::Field(field, subpat.ty));
                    subpats.push(InterPat::lower_thir_pat(cx, subplace, subpat));
                }

                // We treat non-exhaustive enums the same independent of the crate they are
                // defined in, to avoid differences in the operational semantics between crates.
                let refutable =
                    adt_def.variants().len() > 1 || adt_def.is_variant_list_non_exhaustive();
                if refutable {
                    let testable_case = TestableCase::Variant { adt_def, variant_index };
                    InterPatKind::Refutable { place: unwrap_place(), testable_case, subpats }
                } else {
                    InterPatKind::Irrefutable { subpats, binding: None }
                }
            }

            PatKind::Leaf { ref subpatterns } => {
                let mut subpats = vec![];
                for &FieldPat { field, pattern: ref subpat } in subpatterns {
                    let subplace = place_builder.clone_project(PlaceElem::Field(field, subpat.ty));
                    subpats.push(InterPat::lower_thir_pat(cx, subplace, subpat));
                }
                InterPatKind::Irrefutable { subpats, binding: None }
            }

            PatKind::Deref { pin: Pinnedness::Pinned, ref subpattern } => {
                let pinned_ref_ty = match pattern.ty.pinned_ty() {
                    Some(p_ty) if p_ty.is_ref() => p_ty,
                    _ => span_bug!(pattern.span, "bad type for pinned deref: {:?}", pattern.ty),
                };
                let subpat = InterPat::lower_thir_pat(
                    cx,
                    // Project into the `Pin(_)` struct, then deref the inner `&` or `&mut`.
                    place_builder.field(FieldIdx::ZERO, pinned_ref_ty).deref(),
                    subpattern,
                );

                InterPatKind::Irrefutable { subpats: vec![subpat], binding: None }
            }

            PatKind::Deref { pin: Pinnedness::Not, ref subpattern }
            | PatKind::DerefPattern { ref subpattern, borrow: DerefPatBorrowMode::Box } => {
                let subpat = InterPat::lower_thir_pat(cx, place_builder.deref(), subpattern);
                InterPatKind::Irrefutable { subpats: vec![subpat], binding: None }
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
                let subpat =
                    InterPat::lower_thir_pat(cx, PlaceBuilder::from(temp).deref(), subpattern);
                InterPatKind::Refutable {
                    place: unwrap_place(),
                    testable_case: TestableCase::Deref { temp, mutability },
                    subpats: vec![subpat],
                }
            }

            PatKind::Guard { .. } => {
                // FIXME(guard_patterns)
                InterPatKind::Irrefutable { subpats: vec![], binding: None }
            }

            PatKind::Never => InterPatKind::Refutable {
                place: unwrap_place(),
                testable_case: TestableCase::Never,
                subpats: vec![],
            },
        };

        // A pattern node is guaranteed to never match if one of these is true:
        // - The node itself is a never pattern (`!`).
        // - It is not an or-pattern, and one of its subpatterns will never match.
        // - It is an or-pattern, and _all_ of its or-subpatterns will never match.
        let is_never = match &kind {
            InterPatKind::Refutable { testable_case: TestableCase::Never, .. } => true,
            InterPatKind::Refutable { subpats, .. } | InterPatKind::Irrefutable { subpats, .. } => {
                subpats.iter().any(|subpat| subpat.is_never)
            }
            InterPatKind::Or { or_subpats } => or_subpats.iter().all(|subpat| subpat.is_never),
        };

        InterPat { kind, ascriptions, pattern_span: pattern.span, is_never }
    }
}
