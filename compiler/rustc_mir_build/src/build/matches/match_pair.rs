use rustc_middle::mir::*;
use rustc_middle::thir::{self, *};
use rustc_middle::ty::{self, Ty, TypeVisitableExt};

use crate::build::Builder;
use crate::build::expr::as_place::{PlaceBase, PlaceBuilder};
use crate::build::matches::{FlatPat, MatchPairTree, TestCase};

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

        match_pairs.extend(prefix.iter().enumerate().map(|(idx, subpattern)| {
            let elem =
                ProjectionElem::ConstantIndex { offset: idx as u64, min_length, from_end: false };
            MatchPairTree::for_pattern(place.clone_project(elem), subpattern, self)
        }));

        if let Some(subslice_pat) = opt_slice {
            let suffix_len = suffix.len() as u64;
            let subslice = place.clone_project(PlaceElem::Subslice {
                from: prefix.len() as u64,
                to: if exact_size { min_length - suffix_len } else { suffix_len },
                from_end: !exact_size,
            });
            match_pairs.push(MatchPairTree::for_pattern(subslice, subslice_pat, self));
        }

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

impl<'pat, 'tcx> MatchPairTree<'pat, 'tcx> {
    /// Recursively builds a match pair tree for the given pattern and its
    /// subpatterns.
    pub(in crate::build) fn for_pattern(
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

            PatKind::InlineConstant { subpattern: ref pattern, def, .. } => {
                // Apply a type ascription for the inline constant to the value at `match_pair.place`
                let ascription = place.map(|source| {
                    let span = pattern.span;
                    let parent_id = cx.tcx.typeck_root_def_id(cx.def_id.to_def_id());
                    let args = ty::InlineConstArgs::new(cx.tcx, ty::InlineConstArgsParts {
                        parent_args: ty::GenericArgs::identity_for_item(cx.tcx, parent_id),
                        ty: cx.infcx.next_ty_var(span),
                    })
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

                subpairs.push(MatchPairTree::for_pattern(place_builder, pattern, cx));
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
                    i == variant_index
                        || !v
                            .inhabited_predicate(cx.tcx, adt_def)
                            .instantiate(cx.tcx, args)
                            .apply_ignore_module(cx.tcx, cx.param_env)
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

        MatchPairTree { place, test_case, subpairs, pattern, coverage_id: Default::default() }
    }
}
