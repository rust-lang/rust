//! See docs in build/expr/mod.rs

use crate::build::expr::category::Category;
use crate::build::ForGuard::{OutsideGuard, RefWithinGuard};
use crate::build::{BlockAnd, BlockAndExtension, Builder, Capture, CaptureMap};
use rustc_hir::def_id::LocalDefId;
use rustc_middle::hir::place::Projection as HirProjection;
use rustc_middle::hir::place::ProjectionKind as HirProjectionKind;
use rustc_middle::middle::region;
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::mir::AssertKind::BoundsCheck;
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty::AdtDef;
use rustc_middle::ty::{self, CanonicalUserTypeAnnotation, Ty, TyCtxt, Variance};
use rustc_span::Span;
use rustc_target::abi::VariantIdx;

use rustc_index::vec::Idx;

use std::assert_matches::assert_matches;
use std::convert::From;
use std::iter;

/// `PlaceBuilder` is used to create places during MIR construction. It allows you to "build up" a
/// place by pushing more and more projections onto the end, and then convert the final set into a
/// place using the `into_place` method.
///
/// This is used internally when building a place for an expression like `a.b.c`. The fields `b`
/// and `c` can be progressively pushed onto the place builder that is created when converting `a`.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::build) enum PlaceBuilder<'tcx> {
    /// Denotes the start of a `Place`.
    ///
    /// We use `PlaceElem` since this has all `Field` types available.
    Local { local: Local, projection: Vec<PlaceElem<'tcx>> },

    /// When building place for an expression within a closure, the place might start off a
    /// captured path. When `capture_disjoint_fields` is enabled, we might not know the capture
    /// index (within the desugared closure) of the captured path until most of the projections
    /// are applied. We use `PlaceBuilder::Upvar` to keep track of the root variable off of which the
    /// captured path starts, the closure the capture belongs to and the trait the closure
    /// implements.
    ///
    /// Once we have figured out the capture index, we can convert the place builder to
    /// `PlaceBuilder::Local`.
    ///
    /// Consider the following example
    /// ```rust
    /// let t = (((10, 10), 10), 10);
    ///
    /// let c = || {
    ///     println!("{}", t.0.0.0);
    /// };
    /// ```
    /// Here the THIR expression for `t.0.0.0` will be something like
    ///
    /// ```ignore (illustrative)
    /// * Field(0)
    ///     * Field(0)
    ///         * Field(0)
    ///             * UpvarRef(t)
    /// ```
    ///
    /// When `capture_disjoint_fields` is enabled, `t.0.0.0` is captured and we won't be able to
    /// figure out that it is captured until all the `Field` projections are applied.
    ///
    /// Note: in contrast to `PlaceBuilder::Local` we have not yet determined all `Field` types
    /// and will only do so once converting to `PlaceBuilder::Local`.
    Upvar { upvar: Upvar, projection: Vec<UpvarProjectionElem<'tcx>> },
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Upvar {
    var_hir_id: LocalVarId,
    closure_def_id: LocalDefId,
}

/// Given a list of MIR projections, convert them to list of HIR ProjectionKind.
/// The projections are truncated to represent a path that might be captured by a
/// closure/generator. This implies the vector returned from this function doesn't contain
/// ProjectionElems `Downcast`, `ConstantIndex`, `Index`, or `Subslice` because those will never be
/// part of a path that is captured by a closure. We stop applying projections once we see the first
/// projection that isn't captured by a closure.
fn convert_to_hir_projections_and_truncate_for_capture<'tcx>(
    mir_projections: &[UpvarProjectionElem<'tcx>],
) -> Vec<HirProjectionKind> {
    let mut hir_projections = Vec::new();
    let mut variant = None;

    for mir_projection in mir_projections {
        let hir_projection = match mir_projection {
            ProjectionElem::Deref => HirProjectionKind::Deref,
            ProjectionElem::Field(field, _) => {
                let variant = variant.unwrap_or(VariantIdx::new(0));
                HirProjectionKind::Field(field.index() as u32, variant)
            }
            ProjectionElem::Downcast(.., idx) => {
                // We don't expect to see multi-variant enums here, as earlier
                // phases will have truncated them already. However, there can
                // still be downcasts, thanks to single-variant enums.
                // We keep track of VariantIdx so we can use this information
                // if the next ProjectionElem is a Field.
                variant = Some(*idx);
                continue;
            }
            // These do not affect anything, they just make sure we know the right type.
            ProjectionElem::OpaqueCast(_) => continue,
            ProjectionElem::Index(..)
            | ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. } => {
                // We don't capture array-access projections.
                // We can stop here as arrays are captured completely.
                break;
            }
        };
        variant = None;
        hir_projections.push(hir_projection);
    }

    hir_projections
}

/// Return true if the `proj_possible_ancestor` represents an ancestor path
/// to `proj_capture` or `proj_possible_ancestor` is same as `proj_capture`,
/// assuming they both start off of the same root variable.
///
/// **Note:** It's the caller's responsibility to ensure that both lists of projections
///           start off of the same root variable.
///
/// Eg: 1. `foo.x` which is represented using `projections=[Field(x)]` is an ancestor of
///        `foo.x.y` which is represented using `projections=[Field(x), Field(y)]`.
///        Note both `foo.x` and `foo.x.y` start off of the same root variable `foo`.
///     2. Since we only look at the projections here function will return `bar.x` as an a valid
///        ancestor of `foo.x.y`. It's the caller's responsibility to ensure that both projections
///        list are being applied to the same root variable.
fn is_ancestor_or_same_capture(
    proj_possible_ancestor: &[HirProjectionKind],
    proj_capture: &[HirProjectionKind],
) -> bool {
    // We want to make sure `is_ancestor_or_same_capture("x.0.0", "x.0")` to return false.
    // Therefore we can't just check if all projections are same in the zipped iterator below.
    if proj_possible_ancestor.len() > proj_capture.len() {
        return false;
    }

    iter::zip(proj_possible_ancestor, proj_capture).all(|(a, b)| a == b)
}

/// Given a closure, returns the index of a capture within the desugared closure struct and the
/// `ty::CapturedPlace` which is the ancestor of the Place represented using the `var_hir_id`
/// and `projection`.
///
/// Note there will be at most one ancestor for any given Place.
///
/// Returns None, when the ancestor is not found.
fn find_capture_matching_projections<'a, 'tcx>(
    upvars: &'a CaptureMap<'tcx>,
    var_hir_id: LocalVarId,
    projections: &[UpvarProjectionElem<'tcx>],
) -> Option<(usize, &'a Capture<'tcx>)> {
    let hir_projections = convert_to_hir_projections_and_truncate_for_capture(projections);

    upvars.get_by_key_enumerated(var_hir_id.0).find(|(_, capture)| {
        let possible_ancestor_proj_kinds: Vec<_> =
            capture.captured_place.place.projections.iter().map(|proj| proj.kind).collect();
        is_ancestor_or_same_capture(&possible_ancestor_proj_kinds, &hir_projections)
    })
}

/// Takes an upvar place and tries to resolve it into a `PlaceBuilder`
/// with `PlaceBase::Local`
#[instrument(level = "trace", skip(cx), ret)]
fn to_upvars_resolved_place_builder<'tcx>(
    cx: &Builder<'_, 'tcx>,
    var_hir_id: LocalVarId,
    closure_def_id: LocalDefId,
    projection: &[UpvarProjectionElem<'tcx>],
) -> Option<PlaceBuilder<'tcx>> {
    let Some((capture_index, capture)) =
        find_capture_matching_projections(
            &cx.upvars,
            var_hir_id,
            &projection,
        ) else {
        let closure_span = cx.tcx.def_span(closure_def_id);
        if !enable_precise_capture(cx.tcx, closure_span) {
            bug!(
                "No associated capture found for {:?}[{:#?}] even though \
                    capture_disjoint_fields isn't enabled",
                var_hir_id,
                projection
            )
        } else {
            debug!(
                "No associated capture found for {:?}[{:#?}]",
                var_hir_id, projection,
            );
        }

        return None;
    };

    // Access the capture by accessing the field within the Closure struct.
    let capture_info = &cx.upvars[capture_index];

    let Place { local: upvar_resolved_local, projection: local_projection } =
        capture_info.use_place;

    // We used some of the projections to build the capture itself,
    // now we apply the remaining to the upvar resolved place.
    let upvar_projection = strip_prefix(
        capture.captured_place.place.base_ty,
        projection,
        &capture.captured_place.place.projections,
    );

    let upvar_resolved_place_builder = PlaceBuilder::construct_local_place_builder(
        cx,
        upvar_resolved_local,
        local_projection.as_slice(),
        upvar_projection,
    );

    assert!(matches!(upvar_resolved_place_builder, PlaceBuilder::Local { .. }));

    Some(upvar_resolved_place_builder)
}

/// Returns projections remaining after stripping an initial prefix of HIR
/// projections.
///
/// Supports only HIR projection kinds that represent a path that might be
/// captured by a closure or a generator, i.e., an `Index` or a `Subslice`
/// projection kinds are unsupported.
fn strip_prefix<'a, 'tcx>(
    mut base_ty: Ty<'tcx>,
    projections: &'a [UpvarProjectionElem<'tcx>],
    prefix_projections: &[HirProjection<'tcx>],
) -> impl Iterator<Item = UpvarProjectionElem<'tcx>> + 'a {
    let mut iter = projections
        .iter()
        .copied()
        // Filter out opaque casts, they are unnecessary in the prefix.
        .filter(|elem| !matches!(elem, ProjectionElem::OpaqueCast(..)));
    for projection in prefix_projections {
        debug!(?projection, ?projection.ty);

        match projection.kind {
            HirProjectionKind::Deref => {
                assert_matches!(iter.next(), Some(ProjectionElem::Deref));
            }
            HirProjectionKind::Field(..) => {
                if base_ty.is_enum() {
                    assert_matches!(iter.next(), Some(ProjectionElem::Downcast(..)));
                }
                assert_matches!(iter.next(), Some(ProjectionElem::Field(..)));
            }
            HirProjectionKind::Index | HirProjectionKind::Subslice => {
                bug!("unexpected projection kind: {:?}", projection);
            }
        }

        base_ty = projection.ty;
    }

    iter
}

impl<'tcx> PlaceBuilder<'tcx> {
    pub(in crate::build) fn to_place(&self, cx: &Builder<'_, 'tcx>) -> Place<'tcx> {
        self.try_to_place(cx).unwrap()
    }

    /// Creates a `Place` or returns `None` if an upvar cannot be resolved
    pub(in crate::build) fn try_to_place(&self, cx: &Builder<'_, 'tcx>) -> Option<Place<'tcx>> {
        let resolved = self.resolve_upvar(cx);
        let builder = resolved.as_ref().unwrap_or(self);
        let PlaceBuilder::Local{local, ref projection} = builder else { return None };
        let projection = cx.tcx.intern_place_elems(projection);
        Some(Place { local: *local, projection })
    }

    /// Attempts to resolve the `PlaceBuilder`.
    /// Returns `None` if this is not an upvar.
    ///
    /// Upvars resolve may fail for a `PlaceBuilder` when attempting to
    /// resolve a disjoint field whose root variable is not captured
    /// (destructured assignments) or when attempting to resolve a root
    /// variable (discriminant matching with only wildcard arm) that is
    /// not captured. This can happen because the final mir that will be
    /// generated doesn't require a read for this place. Failures will only
    /// happen inside closures.
    pub(in crate::build) fn resolve_upvar(
        &self,
        cx: &Builder<'_, 'tcx>,
    ) -> Option<PlaceBuilder<'tcx>> {
        let PlaceBuilder::Upvar{ upvar: Upvar {var_hir_id, closure_def_id }, projection} = self else {
            return None;
        };

        to_upvars_resolved_place_builder(cx, *var_hir_id, *closure_def_id, &projection)
    }

    #[instrument(skip(cx), level = "debug")]
    pub(crate) fn field(self, cx: &Builder<'_, 'tcx>, f: Field) -> Self {
        match self.clone() {
            PlaceBuilder::Local { local, projection } => {
                let base_place = PlaceBuilder::Local { local, projection };
                let PlaceTy { ty, variant_index } =
                    base_place.to_place(cx).ty(&cx.local_decls, cx.tcx);
                let base_ty = cx.tcx.normalize_erasing_regions(cx.param_env, ty);

                let field_ty = PlaceBuilder::compute_field_ty(cx, f, base_ty, variant_index);

                self.project(ProjectionElem::Field(f, field_ty))
            }
            PlaceBuilder::Upvar { upvar, mut projection } => {
                projection.push(ProjectionElem::Field(f, ()));
                PlaceBuilder::Upvar { upvar, projection }
            }
        }
    }

    pub(crate) fn deref(self) -> Self {
        self.project(PlaceElem::Deref)
    }

    pub(crate) fn downcast(self, adt_def: AdtDef<'tcx>, variant_index: VariantIdx) -> Self {
        self.project(PlaceElem::Downcast(Some(adt_def.variant(variant_index).name), variant_index))
    }

    fn index(self, index: Local) -> Self {
        self.project(PlaceElem::Index(index))
    }

    #[instrument(level = "debug")]
    pub(crate) fn project(self, elem: PlaceElem<'tcx>) -> Self {
        let result = match self {
            PlaceBuilder::Local { local, mut projection } => {
                projection.push(elem);
                PlaceBuilder::Local { local, projection }
            }
            PlaceBuilder::Upvar { upvar, mut projection } => {
                projection.push(elem.into());
                PlaceBuilder::Upvar { upvar, projection }
            }
        };

        debug!(?result);
        result
    }

    /// Same as `.clone().project(..)` but more efficient
    pub(crate) fn clone_project(&self, elem: PlaceElem<'tcx>) -> Self {
        match self {
            PlaceBuilder::Local { local, projection } => PlaceBuilder::Local {
                local: *local,
                projection: Vec::from_iter(projection.iter().copied().chain([elem.into()])),
            },
            PlaceBuilder::Upvar { upvar, projection } => PlaceBuilder::Upvar {
                upvar: *upvar,
                projection: Vec::from_iter(projection.iter().copied().chain([elem.into()])),
            },
        }
    }

    /// Similar to `Place::ty` but needed during mir building.
    ///
    /// Applies the projections in the `PlaceBuilder` to the base
    /// type.
    ///
    /// Fallible as the root of this place may be an upvar for
    /// which no base type can be determined.
    #[instrument(skip(cx), level = "debug")]
    fn compute_field_ty(
        cx: &Builder<'_, 'tcx>,
        field: Field,
        base_ty: Ty<'tcx>,
        variant_index: Option<VariantIdx>,
    ) -> Ty<'tcx> {
        let field_idx = field.as_usize();
        let field_ty = match base_ty.kind() {
            ty::Adt(adt_def, substs) if adt_def.is_enum() => {
                let variant_idx = variant_index.unwrap();
                adt_def.variant(variant_idx).fields[field_idx].ty(cx.tcx, substs)
            }
            ty::Adt(adt_def, substs) => adt_def
                .all_fields()
                .nth(field_idx)
                .unwrap_or_else(|| {
                    bug!(
                        "expected to take field with idx {:?} of fields of {:?}",
                        field_idx,
                        adt_def
                    )
                })
                .ty(cx.tcx, substs),
            ty::Tuple(elems) => elems.iter().nth(field_idx).unwrap_or_else(|| {
                bug!("expected to take field with idx {:?} of {:?}", field_idx, elems)
            }),
            ty::Closure(_, substs) => {
                let substs = substs.as_closure();
                let Some(f_ty) = substs.upvar_tys().nth(field_idx) else {
                    bug!("expected to take field with idx {:?} of {:?}", field_idx, substs.upvar_tys().collect::<Vec<_>>());
                };

                f_ty
            }
            &ty::Generator(def_id, substs, _) => {
                if let Some(var) = variant_index {
                    let gen_body = cx.tcx.optimized_mir(def_id);
                    let Some(layout) = gen_body.generator_layout() else {
                        bug!("No generator layout for {:?}", base_ty);
                    };

                    let Some(&local) = layout.variant_fields[var].get(field) else {
                        bug!("expected to take field {:?} of {:?}", field, layout.variant_fields[var]);
                    };

                    let Some(&f_ty) = layout.field_tys.get(local) else {
                        bug!("expected to get element for {:?} in {:?}", local, layout.field_tys);
                    };

                    f_ty
                } else {
                    let Some(f_ty) = substs.as_generator().prefix_tys().nth(field.index()) else {
                        bug!(
                            "expected to take index {:?} in {:?}",
                            field.index(),
                            substs.as_generator().prefix_tys().collect::<Vec<_>>()
                        );
                    };

                    f_ty
                }
            }
            _ => bug!("couldn't create field type, unexpected base type: {:?}", base_ty),
        };

        cx.tcx.normalize_erasing_regions(cx.param_env, field_ty)
    }

    /// Creates a `PlaceBuilder::Local` from a `PlaceBuilder::Upvar` whose upvars
    /// are resolved. This function takes two kinds of projections: `local_projection`
    /// contains the projections of the captured upvar and `upvar_projection` the
    /// projections that are applied to the captured upvar. The main purpose of this
    /// function is to figure out the `Ty`s of the field projections in `upvar_projection`.
    #[instrument(skip(cx, local, upvar_projection))]
    fn construct_local_place_builder(
        cx: &Builder<'_, 'tcx>,
        local: Local,
        local_projection: &[PlaceElem<'tcx>],
        upvar_projection: impl Iterator<Item = UpvarProjectionElem<'tcx>>,
    ) -> Self {
        // We maintain a `Ty` to which we apply a projection in each iteration over `upvar_projection`.
        // This `ancestor_ty` let's us infer the field type whenever we encounter a
        // `ProjectionElem::Field`.
        let (mut ancestor_ty, mut opt_variant_idx) =
            local_projections_to_ty(cx, local, local_projection);

        // We add all projection elements we encounter to this `Vec`.
        let mut local_projection = local_projection.to_vec();

        for (i, proj) in upvar_projection.enumerate() {
            debug!("i: {:?}, proj: {:?}, local_projection: {:?}", i, proj, local_projection);
            match proj {
                ProjectionElem::Field(field, _) => {
                    let field_ty =
                        PlaceBuilder::compute_field_ty(cx, field, ancestor_ty, opt_variant_idx);
                    debug!(?field_ty);

                    local_projection.push(ProjectionElem::Field(field, field_ty));
                    ancestor_ty = field_ty;
                    opt_variant_idx = None;
                }
                _ => {
                    let proj = upvar_proj_to_place_elem_no_field_proj(proj);
                    (ancestor_ty, opt_variant_idx) = project_ty(cx.tcx, ancestor_ty, proj);
                    local_projection.push(proj);
                }
            }
        }

        PlaceBuilder::Local { local, projection: local_projection }
    }
}

impl<'tcx> From<Local> for PlaceBuilder<'tcx> {
    fn from(local: Local) -> Self {
        Self::Local { local, projection: Vec::new() }
    }
}

impl<'tcx> From<Place<'tcx>> for PlaceBuilder<'tcx> {
    fn from(p: Place<'tcx>) -> Self {
        Self::Local { local: p.local, projection: p.projection.to_vec() }
    }
}

fn project_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    elem: PlaceElem<'tcx>,
) -> (Ty<'tcx>, Option<VariantIdx>) {
    match elem {
        ProjectionElem::Deref => {
            let updated_ty = ty
                .builtin_deref(true)
                .unwrap_or_else(|| bug!("deref projection of non-dereferenceable ty {:?}", ty))
                .ty;

            (updated_ty, None)
        }
        ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } => {
            (ty.builtin_index().unwrap(), None)
        }
        ProjectionElem::Subslice { from, to, from_end } => {
            let ty = match ty.kind() {
                ty::Slice(..) => ty,
                ty::Array(inner, _) if !from_end => tcx.mk_array(*inner, (to - from) as u64),
                ty::Array(inner, size) if from_end => {
                    let size = size.eval_usize(tcx, ty::ParamEnv::empty());
                    let len = size - (from as u64) - (to as u64);
                    tcx.mk_array(*inner, len)
                }
                _ => bug!("cannot subslice non-array type: `{:?}`", ty),
            };

            (ty, None)
        }
        ProjectionElem::Downcast(_, variant_idx) => (ty, Some(variant_idx)),
        ProjectionElem::Field(_, ty) => (ty, None),
        ProjectionElem::OpaqueCast(..) => bug!("didn't expect OpaqueCast"),
    }
}

fn local_projections_to_ty<'a, 'tcx>(
    cx: &'a Builder<'a, 'tcx>,
    local: Local,
    projection: &'a [PlaceElem<'tcx>],
) -> (Ty<'tcx>, Option<VariantIdx>) {
    let local_ty = cx.local_decls.local_decls()[local].ty;
    projection.iter().fold((local_ty, None), |ty_variant_idx, elem| {
        let ty = ty_variant_idx.0;
        project_ty(cx.tcx, ty, *elem)
    })
}

// Converts an `UpvarProjectionElem` to `PlaceElem`, ICE'ing when being passed a
// field projection.
fn upvar_proj_to_place_elem_no_field_proj<'tcx>(
    upvar_proj: UpvarProjectionElem<'tcx>,
) -> PlaceElem<'tcx> {
    match upvar_proj {
        ProjectionElem::Deref => ProjectionElem::Deref,
        ProjectionElem::Index(i) => ProjectionElem::Index(i),
        ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
            ProjectionElem::ConstantIndex { offset, min_length, from_end }
        }
        ProjectionElem::Subslice { from, to, from_end } => {
            ProjectionElem::Subslice { from, to, from_end }
        }
        ProjectionElem::Downcast(ty, variant_idx) => ProjectionElem::Downcast(ty, variant_idx),
        ProjectionElem::OpaqueCast(ty) => ProjectionElem::OpaqueCast(ty),
        ProjectionElem::Field(..) => bug!("should not be called with `ProjectionElem::Field`"),
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Compile `expr`, yielding a place that we can move from etc.
    ///
    /// WARNING: Any user code might:
    /// * Invalidate any slice bounds checks performed.
    /// * Change the address that this `Place` refers to.
    /// * Modify the memory that this place refers to.
    /// * Invalidate the memory that this place refers to, this will be caught
    ///   by borrow checking.
    ///
    /// Extra care is needed if any user code is allowed to run between calling
    /// this method and using it, as is the case for `match` and index
    /// expressions.
    pub(crate) fn as_place(
        &mut self,
        mut block: BasicBlock,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<Place<'tcx>> {
        let place_builder = unpack!(block = self.as_place_builder(block, expr));
        block.and(place_builder.to_place(self))
    }

    /// This is used when constructing a compound `Place`, so that we can avoid creating
    /// intermediate `Place` values until we know the full set of projections.
    pub(crate) fn as_place_builder(
        &mut self,
        block: BasicBlock,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<PlaceBuilder<'tcx>> {
        self.expr_as_place(block, expr, Mutability::Mut, None)
    }

    /// Compile `expr`, yielding a place that we can move from etc.
    /// Mutability note: The caller of this method promises only to read from the resulting
    /// place. The place itself may or may not be mutable:
    /// * If this expr is a place expr like a.b, then we will return that place.
    /// * Otherwise, a temporary is created: in that event, it will be an immutable temporary.
    pub(crate) fn as_read_only_place(
        &mut self,
        mut block: BasicBlock,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<Place<'tcx>> {
        let place_builder = unpack!(block = self.as_read_only_place_builder(block, expr));
        block.and(place_builder.to_place(self))
    }

    /// This is used when constructing a compound `Place`, so that we can avoid creating
    /// intermediate `Place` values until we know the full set of projections.
    /// Mutability note: The caller of this method promises only to read from the resulting
    /// place. The place itself may or may not be mutable:
    /// * If this expr is a place expr like a.b, then we will return that place.
    /// * Otherwise, a temporary is created: in that event, it will be an immutable temporary.
    fn as_read_only_place_builder(
        &mut self,
        block: BasicBlock,
        expr: &Expr<'tcx>,
    ) -> BlockAnd<PlaceBuilder<'tcx>> {
        self.expr_as_place(block, expr, Mutability::Not, None)
    }

    #[instrument(skip(self, fake_borrow_temps), level = "debug")]
    fn expr_as_place(
        &mut self,
        mut block: BasicBlock,
        expr: &Expr<'tcx>,
        mutability: Mutability,
        fake_borrow_temps: Option<&mut Vec<Local>>,
    ) -> BlockAnd<PlaceBuilder<'tcx>> {
        let this = self;
        let expr_span = expr.span;
        let source_info = this.source_info(expr_span);
        match expr.kind {
            ExprKind::Scope { region_scope, lint_level, value } => {
                this.in_scope((region_scope, source_info), lint_level, |this| {
                    this.expr_as_place(block, &this.thir[value], mutability, fake_borrow_temps)
                })
            }
            ExprKind::Field { lhs, variant_index, name } => {
                let lhs = &this.thir[lhs];
                let mut place_builder =
                    unpack!(block = this.expr_as_place(block, lhs, mutability, fake_borrow_temps,));
                debug!(?place_builder);
                if let ty::Adt(adt_def, _) = lhs.ty.kind() {
                    if adt_def.is_enum() {
                        place_builder = place_builder.downcast(*adt_def, variant_index);
                    }
                }
                block.and(place_builder.field(this, name))
            }
            ExprKind::Deref { arg } => {
                let place_builder = unpack!(
                    block =
                        this.expr_as_place(block, &this.thir[arg], mutability, fake_borrow_temps,)
                );
                block.and(place_builder.deref())
            }
            ExprKind::Index { lhs, index } => this.lower_index_expression(
                block,
                &this.thir[lhs],
                &this.thir[index],
                mutability,
                fake_borrow_temps,
                expr.temp_lifetime,
                expr_span,
                source_info,
            ),
            ExprKind::UpvarRef { closure_def_id, var_hir_id } => {
                this.lower_captured_upvar(block, closure_def_id.expect_local(), var_hir_id)
            }

            ExprKind::VarRef { id } => {
                let place_builder = if this.is_bound_var_in_guard(id) {
                    let index = this.var_local_id(id, RefWithinGuard);
                    PlaceBuilder::from(index).deref()
                } else {
                    let index = this.var_local_id(id, OutsideGuard);
                    PlaceBuilder::from(index)
                };
                block.and(place_builder)
            }

            ExprKind::PlaceTypeAscription { source, ref user_ty } => {
                let place_builder = unpack!(
                    block = this.expr_as_place(
                        block,
                        &this.thir[source],
                        mutability,
                        fake_borrow_temps,
                    )
                );
                if let Some(user_ty) = user_ty {
                    let annotation_index =
                        this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                            span: source_info.span,
                            user_ty: user_ty.clone(),
                            inferred_ty: expr.ty,
                        });

                    let place = place_builder.to_place(this);
                    this.cfg.push(
                        block,
                        Statement {
                            source_info,
                            kind: StatementKind::AscribeUserType(
                                Box::new((
                                    place,
                                    UserTypeProjection { base: annotation_index, projs: vec![] },
                                )),
                                Variance::Invariant,
                            ),
                        },
                    );
                }
                block.and(place_builder)
            }
            ExprKind::ValueTypeAscription { source, ref user_ty } => {
                let source = &this.thir[source];
                let temp =
                    unpack!(block = this.as_temp(block, source.temp_lifetime, source, mutability));
                if let Some(user_ty) = user_ty {
                    let annotation_index =
                        this.canonical_user_type_annotations.push(CanonicalUserTypeAnnotation {
                            span: source_info.span,
                            user_ty: user_ty.clone(),
                            inferred_ty: expr.ty,
                        });
                    this.cfg.push(
                        block,
                        Statement {
                            source_info,
                            kind: StatementKind::AscribeUserType(
                                Box::new((
                                    Place::from(temp),
                                    UserTypeProjection { base: annotation_index, projs: vec![] },
                                )),
                                Variance::Invariant,
                            ),
                        },
                    );
                }
                block.and(PlaceBuilder::from(temp))
            }

            ExprKind::Array { .. }
            | ExprKind::Tuple { .. }
            | ExprKind::Adt { .. }
            | ExprKind::Closure { .. }
            | ExprKind::Unary { .. }
            | ExprKind::Binary { .. }
            | ExprKind::LogicalOp { .. }
            | ExprKind::Box { .. }
            | ExprKind::Cast { .. }
            | ExprKind::Use { .. }
            | ExprKind::NeverToAny { .. }
            | ExprKind::Pointer { .. }
            | ExprKind::Repeat { .. }
            | ExprKind::Borrow { .. }
            | ExprKind::AddressOf { .. }
            | ExprKind::Match { .. }
            | ExprKind::If { .. }
            | ExprKind::Loop { .. }
            | ExprKind::Block { .. }
            | ExprKind::Let { .. }
            | ExprKind::Assign { .. }
            | ExprKind::AssignOp { .. }
            | ExprKind::Break { .. }
            | ExprKind::Continue { .. }
            | ExprKind::Return { .. }
            | ExprKind::Literal { .. }
            | ExprKind::NamedConst { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ZstLiteral { .. }
            | ExprKind::ConstParam { .. }
            | ExprKind::ConstBlock { .. }
            | ExprKind::StaticRef { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::Yield { .. }
            | ExprKind::ThreadLocalRef(_)
            | ExprKind::Call { .. } => {
                // these are not places, so we need to make a temporary.
                debug_assert!(!matches!(Category::of(&expr.kind), Some(Category::Place)));
                let temp =
                    unpack!(block = this.as_temp(block, expr.temp_lifetime, expr, mutability));
                block.and(PlaceBuilder::from(temp))
            }
        }
    }

    /// Lower a captured upvar. Note we might not know the actual capture index,
    /// so we create a place starting from `Upvar`, which will be resolved
    /// once all projections that allow us to identify a capture have been applied.
    fn lower_captured_upvar(
        &mut self,
        block: BasicBlock,
        closure_def_id: LocalDefId,
        var_hir_id: LocalVarId,
    ) -> BlockAnd<PlaceBuilder<'tcx>> {
        block.and(PlaceBuilder::Upvar {
            upvar: Upvar { var_hir_id, closure_def_id },
            projection: vec![],
        })
    }

    /// Lower an index expression
    ///
    /// This has two complications;
    ///
    /// * We need to do a bounds check.
    /// * We need to ensure that the bounds check can't be invalidated using an
    ///   expression like `x[1][{x = y; 2}]`. We use fake borrows here to ensure
    ///   that this is the case.
    fn lower_index_expression(
        &mut self,
        mut block: BasicBlock,
        base: &Expr<'tcx>,
        index: &Expr<'tcx>,
        mutability: Mutability,
        fake_borrow_temps: Option<&mut Vec<Local>>,
        temp_lifetime: Option<region::Scope>,
        expr_span: Span,
        source_info: SourceInfo,
    ) -> BlockAnd<PlaceBuilder<'tcx>> {
        let base_fake_borrow_temps = &mut Vec::new();
        let is_outermost_index = fake_borrow_temps.is_none();
        let fake_borrow_temps = fake_borrow_temps.unwrap_or(base_fake_borrow_temps);

        let base_place =
            unpack!(block = self.expr_as_place(block, base, mutability, Some(fake_borrow_temps),));

        // Making this a *fresh* temporary means we do not have to worry about
        // the index changing later: Nothing will ever change this temporary.
        // The "retagging" transformation (for Stacked Borrows) relies on this.
        let idx = unpack!(block = self.as_temp(block, temp_lifetime, index, Mutability::Not,));

        block = self.bounds_check(block, &base_place, idx, expr_span, source_info);

        if is_outermost_index {
            self.read_fake_borrows(block, fake_borrow_temps, source_info)
        } else {
            self.add_fake_borrows_of_base(
                base_place.to_place(self),
                block,
                fake_borrow_temps,
                expr_span,
                source_info,
            );
        }

        block.and(base_place.index(idx))
    }

    fn bounds_check(
        &mut self,
        block: BasicBlock,
        slice: &PlaceBuilder<'tcx>,
        index: Local,
        expr_span: Span,
        source_info: SourceInfo,
    ) -> BasicBlock {
        let usize_ty = self.tcx.types.usize;
        let bool_ty = self.tcx.types.bool;
        // bounds check:
        let len = self.temp(usize_ty, expr_span);
        let lt = self.temp(bool_ty, expr_span);

        // len = len(slice)
        self.cfg.push_assign(block, source_info, len, Rvalue::Len(slice.to_place(self)));
        // lt = idx < len
        self.cfg.push_assign(
            block,
            source_info,
            lt,
            Rvalue::BinaryOp(
                BinOp::Lt,
                Box::new((Operand::Copy(Place::from(index)), Operand::Copy(len))),
            ),
        );
        let msg = BoundsCheck { len: Operand::Move(len), index: Operand::Copy(Place::from(index)) };
        // assert!(lt, "...")
        self.assert(block, Operand::Move(lt), true, msg, expr_span)
    }

    fn add_fake_borrows_of_base(
        &mut self,
        base_place: Place<'tcx>,
        block: BasicBlock,
        fake_borrow_temps: &mut Vec<Local>,
        expr_span: Span,
        source_info: SourceInfo,
    ) {
        let tcx = self.tcx;
        let place_ty = base_place.ty(&self.local_decls, tcx);

        if let ty::Slice(_) = place_ty.ty.kind() {
            // We need to create fake borrows to ensure that the bounds
            // check that we just did stays valid. Since we can't assign to
            // unsized values, we only need to ensure that none of the
            // pointers in the base place are modified.
            for (idx, elem) in base_place.projection.iter().enumerate().rev() {
                match elem {
                    ProjectionElem::Deref => {
                        let fake_borrow_deref_ty = Place::ty_from(
                            base_place.local,
                            &base_place.projection[..idx],
                            &self.local_decls,
                            tcx,
                        )
                        .ty;
                        let fake_borrow_ty =
                            tcx.mk_imm_ref(tcx.lifetimes.re_erased, fake_borrow_deref_ty);
                        let fake_borrow_temp =
                            self.local_decls.push(LocalDecl::new(fake_borrow_ty, expr_span));
                        let projection = tcx.intern_place_elems(&base_place.projection[..idx]);
                        self.cfg.push_assign(
                            block,
                            source_info,
                            fake_borrow_temp.into(),
                            Rvalue::Ref(
                                tcx.lifetimes.re_erased,
                                BorrowKind::Shallow,
                                Place { local: base_place.local, projection },
                            ),
                        );
                        fake_borrow_temps.push(fake_borrow_temp);
                    }
                    ProjectionElem::Index(_) => {
                        let index_ty = Place::ty_from(
                            base_place.local,
                            &base_place.projection[..idx],
                            &self.local_decls,
                            tcx,
                        );
                        match index_ty.ty.kind() {
                            // The previous index expression has already
                            // done any index expressions needed here.
                            ty::Slice(_) => break,
                            ty::Array(..) => (),
                            _ => bug!("unexpected index base"),
                        }
                    }
                    ProjectionElem::Field(..)
                    | ProjectionElem::Downcast(..)
                    | ProjectionElem::OpaqueCast(..)
                    | ProjectionElem::ConstantIndex { .. }
                    | ProjectionElem::Subslice { .. } => (),
                }
            }
        }
    }

    fn read_fake_borrows(
        &mut self,
        bb: BasicBlock,
        fake_borrow_temps: &mut Vec<Local>,
        source_info: SourceInfo,
    ) {
        // All indexes have been evaluated now, read all of the
        // fake borrows so that they are live across those index
        // expressions.
        for temp in fake_borrow_temps {
            self.cfg.push_fake_read(bb, source_info, FakeReadCause::ForIndex, Place::from(*temp));
        }
    }
}

/// Precise capture is enabled if the feature gate `capture_disjoint_fields` is enabled or if
/// user is using Rust Edition 2021 or higher.
fn enable_precise_capture(tcx: TyCtxt<'_>, closure_span: Span) -> bool {
    tcx.features().capture_disjoint_fields || closure_span.rust_2021()
}
