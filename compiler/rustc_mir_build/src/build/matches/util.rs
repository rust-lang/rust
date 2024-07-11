use std::marker::PhantomData;

use crate::build::expr::as_place::{PlaceBase, PlaceBuilder};
use crate::build::matches::{Binding, Candidate, FlatPat, MatchPair, TestCase};
use crate::build::Builder;
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::*;
use rustc_middle::thir::{self, *};
use rustc_middle::ty::TypeVisitableExt;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use tracing::debug;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    pub(crate) fn field_match_pairs<'pat>(
        &mut self,
        place: PlaceBuilder<'tcx>,
        subpatterns: &'pat [FieldPat<'tcx>],
    ) -> Vec<MatchPair<'pat, 'tcx>> {
        subpatterns
            .iter()
            .map(|fieldpat| {
                let place =
                    place.clone_project(PlaceElem::Field(fieldpat.field, fieldpat.pattern.ty));
                MatchPair::new(place, &fieldpat.pattern, self)
            })
            .collect()
    }

    pub(crate) fn prefix_slice_suffix<'pat>(
        &mut self,
        match_pairs: &mut Vec<MatchPair<'pat, 'tcx>>,
        place: &PlaceBuilder<'tcx>,
        prefix: &'pat [Box<Pat<'tcx>>],
        opt_slice: &'pat Option<Box<Pat<'tcx>>>,
        suffix: &'pat [Box<Pat<'tcx>>],
    ) {
        let tcx = self.tcx;
        let (min_length, exact_size) = if let Some(place_resolved) = place.try_to_place(self) {
            match place_resolved.ty(&self.local_decls, tcx).ty.kind() {
                ty::Array(_, length) => (length.eval_target_usize(tcx, self.param_env), true),
                _ => ((prefix.len() + suffix.len()).try_into().unwrap(), false),
            }
        } else {
            ((prefix.len() + suffix.len()).try_into().unwrap(), false)
        };

        match_pairs.extend(prefix.iter().enumerate().map(|(idx, subpattern)| {
            let elem =
                ProjectionElem::ConstantIndex { offset: idx as u64, min_length, from_end: false };
            MatchPair::new(place.clone_project(elem), subpattern, self)
        }));

        if let Some(subslice_pat) = opt_slice {
            let suffix_len = suffix.len() as u64;
            let subslice = place.clone_project(PlaceElem::Subslice {
                from: prefix.len() as u64,
                to: if exact_size { min_length - suffix_len } else { suffix_len },
                from_end: !exact_size,
            });
            match_pairs.push(MatchPair::new(subslice, subslice_pat, self));
        }

        match_pairs.extend(suffix.iter().rev().enumerate().map(|(idx, subpattern)| {
            let end_offset = (idx + 1) as u64;
            let elem = ProjectionElem::ConstantIndex {
                offset: if exact_size { min_length - end_offset } else { end_offset },
                min_length,
                from_end: !exact_size,
            };
            let place = place.clone_project(elem);
            MatchPair::new(place, subpattern, self)
        }));
    }

    /// Creates a false edge to `imaginary_target` and a real edge to
    /// real_target. If `imaginary_target` is none, or is the same as the real
    /// target, a Goto is generated instead to simplify the generated MIR.
    pub(crate) fn false_edges(
        &mut self,
        from_block: BasicBlock,
        real_target: BasicBlock,
        imaginary_target: Option<BasicBlock>,
        source_info: SourceInfo,
    ) {
        match imaginary_target {
            Some(target) if target != real_target => {
                self.cfg.terminate(
                    from_block,
                    source_info,
                    TerminatorKind::FalseEdge { real_target, imaginary_target: target },
                );
            }
            _ => self.cfg.goto(from_block, source_info, real_target),
        }
    }
}

impl<'pat, 'tcx> MatchPair<'pat, 'tcx> {
    /// Recursively builds a `MatchPair` tree for the given pattern and its
    /// subpatterns.
    pub(in crate::build) fn new(
        mut place_builder: PlaceBuilder<'tcx>,
        pattern: &'pat Pat<'tcx>,
        cx: &mut Builder<'_, 'tcx>,
    ) -> MatchPair<'pat, 'tcx> {
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

                subpairs.push(MatchPair::new(place_builder, subpattern, cx));
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
                    subpairs.push(MatchPair::new(place_builder, subpattern, cx));
                }
                TestCase::Irrefutable { ascription: None, binding }
            }

            PatKind::InlineConstant { subpattern: ref pattern, def, .. } => {
                // Apply a type ascription for the inline constant to the value at `match_pair.place`
                let ascription = place.map(|source| {
                    let span = pattern.span;
                    let parent_id = cx.tcx.typeck_root_def_id(cx.def_id.to_def_id());
                    let args = ty::InlineConstArgs::new(
                        cx.tcx,
                        ty::InlineConstArgsParts {
                            parent_args: ty::GenericArgs::identity_for_item(cx.tcx, parent_id),
                            ty: cx.infcx.next_ty_var(span),
                        },
                    )
                    .args;
                    let user_ty = cx.infcx.canonicalize_user_type_annotation(ty::UserType::TypeOf(
                        def.to_def_id(),
                        ty::UserArgs { args, user_self_ty: None },
                    ));
                    let annotation = ty::CanonicalUserTypeAnnotation {
                        inferred_ty: pattern.ty,
                        span,
                        user_ty: Box::new(user_ty),
                    };
                    super::Ascription { annotation, source, variance: ty::Contravariant }
                });

                subpairs.push(MatchPair::new(place_builder, pattern, cx));
                TestCase::Irrefutable { ascription, binding: None }
            }

            PatKind::Array { ref prefix, ref slice, ref suffix } => {
                cx.prefix_slice_suffix(&mut subpairs, &place_builder, prefix, slice, suffix);
                default_irrefutable()
            }
            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                cx.prefix_slice_suffix(&mut subpairs, &place_builder, prefix, slice, suffix);

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
                    i == variant_index || {
                        (cx.tcx.features().exhaustive_patterns
                            || cx.tcx.features().min_exhaustive_patterns)
                            && !v
                                .inhabited_predicate(cx.tcx, adt_def)
                                .instantiate(cx.tcx, args)
                                .apply_ignore_module(cx.tcx, cx.param_env)
                    }
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
                subpairs.push(MatchPair::new(place_builder.deref(), subpattern, cx));
                default_irrefutable()
            }

            PatKind::DerefPattern { ref subpattern, mutability } => {
                // Create a new temporary for each deref pattern.
                // FIXME(deref_patterns): dedup temporaries to avoid multiple `deref()` calls?
                let temp = cx.temp(
                    Ty::new_ref(cx.tcx, cx.tcx.lifetimes.re_erased, subpattern.ty, mutability),
                    pattern.span,
                );
                subpairs.push(MatchPair::new(PlaceBuilder::from(temp).deref(), subpattern, cx));
                TestCase::Deref { temp, mutability }
            }

            PatKind::Never => TestCase::Never,
        };

        MatchPair { place, test_case, subpairs, pattern }
    }
}

/// Determine the set of places that have to be stable across match guards.
///
/// Returns a list of places that need a fake borrow along with a local to store it.
///
/// Match exhaustiveness checking is not able to handle the case where the place being matched on is
/// mutated in the guards. We add "fake borrows" to the guards that prevent any mutation of the
/// place being matched. There are a some subtleties:
///
/// 1. Borrowing `*x` doesn't prevent assigning to `x`. If `x` is a shared reference, the borrow
///    isn't even tracked. As such we have to add fake borrows of any prefixes of a place.
/// 2. We don't want `match x { (Some(_), _) => (), .. }` to conflict with mutable borrows of `x.1`, so we
///    only add fake borrows for places which are bound or tested by the match.
/// 3. We don't want `match x { Some(_) => (), .. }` to conflict with mutable borrows of `(x as
///    Some).0`, so the borrows are a special shallow borrow that only affects the place and not its
///    projections.
///    ```rust
///    let mut x = (Some(0), true);
///    match x {
///        (Some(_), false) => {}
///        _ if { if let Some(ref mut y) = x.0 { *y += 1 }; true } => {}
///        _ => {}
///    }
///    ```
/// 4. The fake borrows may be of places in inactive variants, e.g. here we need to fake borrow `x`
///    and `(x as Some).0`, but when we reach the guard `x` may not be `Some`.
///    ```rust
///    let mut x = (Some(Some(0)), true);
///    match x {
///        (Some(Some(_)), false) => {}
///        _ if { if let Some(Some(ref mut y)) = x.0 { *y += 1 }; true } => {}
///        _ => {}
///    }
///    ```
///    So it would be UB to generate code for the fake borrows. They therefore have to be removed by
///    a MIR pass run after borrow checking.
pub(super) fn collect_fake_borrows<'tcx>(
    cx: &mut Builder<'_, 'tcx>,
    candidates: &[&mut Candidate<'_, 'tcx>],
    temp_span: Span,
    scrutinee_base: PlaceBase,
) -> Vec<(Place<'tcx>, Local, FakeBorrowKind)> {
    let mut collector =
        FakeBorrowCollector { cx, scrutinee_base, fake_borrows: FxIndexMap::default() };
    for candidate in candidates.iter() {
        collector.visit_candidate(candidate);
    }
    let fake_borrows = collector.fake_borrows;
    debug!("add_fake_borrows fake_borrows = {:?}", fake_borrows);
    let tcx = cx.tcx;
    fake_borrows
        .iter()
        .map(|(matched_place, borrow_kind)| {
            let fake_borrow_deref_ty = matched_place.ty(&cx.local_decls, tcx).ty;
            let fake_borrow_ty =
                Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, fake_borrow_deref_ty);
            let mut fake_borrow_temp = LocalDecl::new(fake_borrow_ty, temp_span);
            fake_borrow_temp.local_info = ClearCrossCrate::Set(Box::new(LocalInfo::FakeBorrow));
            let fake_borrow_temp = cx.local_decls.push(fake_borrow_temp);
            (*matched_place, fake_borrow_temp, *borrow_kind)
        })
        .collect()
}

pub(super) struct FakeBorrowCollector<'a, 'b, 'tcx> {
    cx: &'a mut Builder<'b, 'tcx>,
    /// Base of the scrutinee place. Used to distinguish bindings inside the scrutinee place from
    /// bindings inside deref patterns.
    scrutinee_base: PlaceBase,
    /// Store for each place the kind of borrow to take. In case of conflicts, we take the strongest
    /// borrow (i.e. Deep > Shallow).
    /// Invariant: for any place in `fake_borrows`, all the prefixes of this place that are
    /// dereferences are also borrowed with the same of stronger borrow kind.
    fake_borrows: FxIndexMap<Place<'tcx>, FakeBorrowKind>,
}

impl<'a, 'b, 'tcx> FakeBorrowCollector<'a, 'b, 'tcx> {
    // Fake borrow this place and its dereference prefixes.
    fn fake_borrow(&mut self, place: Place<'tcx>, kind: FakeBorrowKind) {
        if self.fake_borrows.get(&place).is_some_and(|k| *k >= kind) {
            return;
        }
        self.fake_borrows.insert(place, kind);
        // Also fake borrow the prefixes of any fake borrow.
        self.fake_borrow_deref_prefixes(place, kind);
    }

    // Fake borrow the prefixes of this place that are dereferences.
    fn fake_borrow_deref_prefixes(&mut self, place: Place<'tcx>, kind: FakeBorrowKind) {
        for (place_ref, elem) in place.as_ref().iter_projections().rev() {
            if let ProjectionElem::Deref = elem {
                // Insert a shallow borrow after a deref. For other projections the borrow of
                // `place_ref` will conflict with any mutation of `place.base`.
                let place = place_ref.to_place(self.cx.tcx);
                if self.fake_borrows.get(&place).is_some_and(|k| *k >= kind) {
                    return;
                }
                self.fake_borrows.insert(place, kind);
            }
        }
    }

    fn visit_candidate(&mut self, candidate: &Candidate<'_, 'tcx>) {
        for binding in &candidate.extra_data.bindings {
            self.visit_binding(binding);
        }
        for match_pair in &candidate.match_pairs {
            self.visit_match_pair(match_pair);
        }
    }

    fn visit_flat_pat(&mut self, flat_pat: &FlatPat<'_, 'tcx>) {
        for binding in &flat_pat.extra_data.bindings {
            self.visit_binding(binding);
        }
        for match_pair in &flat_pat.match_pairs {
            self.visit_match_pair(match_pair);
        }
    }

    fn visit_match_pair(&mut self, match_pair: &MatchPair<'_, 'tcx>) {
        if let TestCase::Or { pats, .. } = &match_pair.test_case {
            for flat_pat in pats.iter() {
                self.visit_flat_pat(flat_pat)
            }
        } else if matches!(match_pair.test_case, TestCase::Deref { .. }) {
            // The subpairs of a deref pattern are all places relative to the deref temporary, so we
            // don't fake borrow them. Problem is, if we only shallowly fake-borrowed
            // `match_pair.place`, this would allow:
            // ```
            // let mut b = Box::new(false);
            // match b {
            //     deref!(true) => {} // not reached because `*b == false`
            //     _ if { *b = true; false } => {} // not reached because the guard is `false`
            //     deref!(false) => {} // not reached because the guard changed it
            //     // UB because we reached the unreachable.
            // }
            // ```
            // Hence we fake borrow using a deep borrow.
            if let Some(place) = match_pair.place {
                self.fake_borrow(place, FakeBorrowKind::Deep);
            }
        } else {
            // Insert a Shallow borrow of any place that is switched on.
            if let Some(place) = match_pair.place {
                self.fake_borrow(place, FakeBorrowKind::Shallow);
            }

            for subpair in &match_pair.subpairs {
                self.visit_match_pair(subpair);
            }
        }
    }

    fn visit_binding(&mut self, Binding { source, .. }: &Binding<'tcx>) {
        if let PlaceBase::Local(l) = self.scrutinee_base
            && l != source.local
        {
            // The base of this place is a temporary created for deref patterns. We don't emit fake
            // borrows for these as they are not initialized in all branches.
            return;
        }

        // Insert a borrows of prefixes of places that are bound and are
        // behind a dereference projection.
        //
        // These borrows are taken to avoid situations like the following:
        //
        // match x[10] {
        //     _ if { x = &[0]; false } => (),
        //     y => (), // Out of bounds array access!
        // }
        //
        // match *x {
        //     // y is bound by reference in the guard and then by copy in the
        //     // arm, so y is 2 in the arm!
        //     y if { y == 1 && (x = &2) == () } => y,
        //     _ => 3,
        // }
        //
        // We don't just fake borrow the whole place because this is allowed:
        // match u {
        //     _ if { u = true; false } => (),
        //     x => (),
        // }
        self.fake_borrow_deref_prefixes(*source, FakeBorrowKind::Shallow);
    }
}

/// Visit all the bindings of these candidates. Because or-alternatives bind the same variables, we
/// only explore the first one of each or-pattern.
pub(super) fn visit_bindings<'tcx>(
    candidates: &[&mut Candidate<'_, 'tcx>],
    f: impl FnMut(&Binding<'tcx>),
) {
    let mut visitor = BindingsVisitor { f, phantom: PhantomData };
    for candidate in candidates.iter() {
        visitor.visit_candidate(candidate);
    }
}

pub(super) struct BindingsVisitor<'tcx, F> {
    f: F,
    phantom: PhantomData<&'tcx ()>,
}

impl<'tcx, F> BindingsVisitor<'tcx, F>
where
    F: FnMut(&Binding<'tcx>),
{
    fn visit_candidate(&mut self, candidate: &Candidate<'_, 'tcx>) {
        for binding in &candidate.extra_data.bindings {
            (self.f)(binding)
        }
        for match_pair in &candidate.match_pairs {
            self.visit_match_pair(match_pair);
        }
    }

    fn visit_flat_pat(&mut self, flat_pat: &FlatPat<'_, 'tcx>) {
        for binding in &flat_pat.extra_data.bindings {
            (self.f)(binding)
        }
        for match_pair in &flat_pat.match_pairs {
            self.visit_match_pair(match_pair);
        }
    }

    fn visit_match_pair(&mut self, match_pair: &MatchPair<'_, 'tcx>) {
        if let TestCase::Or { pats, .. } = &match_pair.test_case {
            // All the or-alternatives should bind the same locals, so we only visit the first one.
            self.visit_flat_pat(&pats[0])
        } else {
            for subpair in &match_pair.subpairs {
                self.visit_match_pair(subpair);
            }
        }
    }
}

#[must_use]
pub(crate) fn ref_pat_borrow_kind(ref_mutability: Mutability) -> BorrowKind {
    match ref_mutability {
        Mutability::Mut => BorrowKind::Mut { kind: MutBorrowKind::Default },
        Mutability::Not => BorrowKind::Shared,
    }
}
