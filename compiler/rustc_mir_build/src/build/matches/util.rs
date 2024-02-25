use crate::build::expr::as_place::{PlaceBase, PlaceBuilder};
use crate::build::matches::{MatchPair, TestCase};
use crate::build::Builder;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::mir::*;
use rustc_middle::thir::{self, *};
use rustc_middle::ty;
use rustc_middle::ty::TypeVisitableExt;

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
    pub(in crate::build) fn new(
        mut place: PlaceBuilder<'tcx>,
        pattern: &'pat Pat<'tcx>,
        cx: &mut Builder<'_, 'tcx>,
    ) -> MatchPair<'pat, 'tcx> {
        // Force the place type to the pattern's type.
        // FIXME(oli-obk): can we use this to simplify slice/array pattern hacks?
        if let Some(resolved) = place.resolve_upvar(cx) {
            place = resolved;
        }

        // Only add the OpaqueCast projection if the given place is an opaque type and the
        // expected type from the pattern is not.
        let may_need_cast = match place.base() {
            PlaceBase::Local(local) => {
                let ty = Place::ty_from(local, place.projection(), &cx.local_decls, cx.tcx).ty;
                ty != pattern.ty && ty.has_opaque_types()
            }
            _ => true,
        };
        if may_need_cast {
            place = place.project(ProjectionElem::OpaqueCast(pattern.ty));
        }

        let default_irrefutable = || TestCase::Irrefutable { binding: None, ascription: None };
        let mut subpairs = Vec::new();
        let test_case = match pattern.kind {
            PatKind::Never | PatKind::Wild | PatKind::Error(_) => default_irrefutable(),
            PatKind::Or { .. } => TestCase::Or,

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
                let ascription = place.try_to_place(cx).map(|source| super::Ascription {
                    annotation: annotation.clone(),
                    source,
                    variance,
                });

                subpairs.push(MatchPair::new(place.clone(), subpattern, cx));
                TestCase::Irrefutable { ascription, binding: None }
            }

            PatKind::Binding {
                name: _,
                mutability: _,
                mode,
                var,
                ty: _,
                ref subpattern,
                is_primary: _,
            } => {
                let binding = place.try_to_place(cx).map(|source| super::Binding {
                    span: pattern.span,
                    source,
                    var_id: var,
                    binding_mode: mode,
                });

                if let Some(subpattern) = subpattern.as_ref() {
                    // this is the `x @ P` case; have to keep matching against `P` now
                    subpairs.push(MatchPair::new(place.clone(), subpattern, cx));
                }
                TestCase::Irrefutable { ascription: None, binding }
            }

            PatKind::InlineConstant { subpattern: ref pattern, def, .. } => {
                // Apply a type ascription for the inline constant to the value at `match_pair.place`
                let ascription = place.try_to_place(cx).map(|source| {
                    let span = pattern.span;
                    let parent_id = cx.tcx.typeck_root_def_id(cx.def_id.to_def_id());
                    let args = ty::InlineConstArgs::new(
                        cx.tcx,
                        ty::InlineConstArgsParts {
                            parent_args: ty::GenericArgs::identity_for_item(cx.tcx, parent_id),
                            ty: cx.infcx.next_ty_var(TypeVariableOrigin {
                                kind: TypeVariableOriginKind::MiscVariable,
                                span,
                            }),
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

                subpairs.push(MatchPair::new(place.clone(), pattern, cx));
                TestCase::Irrefutable { ascription, binding: None }
            }

            PatKind::Array { ref prefix, ref slice, ref suffix } => {
                cx.prefix_slice_suffix(&mut subpairs, &place, prefix, slice, suffix);
                default_irrefutable()
            }
            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                cx.prefix_slice_suffix(&mut subpairs, &place, prefix, slice, suffix);

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
                let downcast_place = place.clone().downcast(adt_def, variant_index); // `(x as Variant)`
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
                subpairs = cx.field_match_pairs(place.clone(), subpatterns);
                default_irrefutable()
            }

            PatKind::Deref { ref subpattern } => {
                let place_builder = place.clone().deref();
                subpairs.push(MatchPair::new(place_builder, subpattern, cx));
                default_irrefutable()
            }
        };

        MatchPair { place, test_case, subpairs, pattern }
    }
}
