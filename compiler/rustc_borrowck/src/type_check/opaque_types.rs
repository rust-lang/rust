use std::iter;

use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::span_bug;
use rustc_middle::ty::{
    self, GenericArgKind, OpaqueHiddenType, OpaqueTypeKey, Ty, TyCtxt, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor, fold_regions,
};
use tracing::{debug, trace};

use super::{MemberConstraintSet, TypeChecker};

/// Once we're done with typechecking the body, we take all the opaque types
/// defined by this function and add their 'member constraints'.
pub(super) fn take_opaques_and_register_member_constraints<'tcx>(
    typeck: &mut TypeChecker<'_, 'tcx>,
) -> FxIndexMap<OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>> {
    let infcx = typeck.infcx;
    // Annoying: to invoke `typeck.to_region_vid`, we need access to
    // `typeck.constraints`, but we also want to be mutating
    // `typeck.member_constraints`. For now, just swap out the value
    // we want and replace at the end.
    let mut member_constraints = std::mem::take(&mut typeck.constraints.member_constraints);
    let opaque_types = infcx
        .take_opaque_types()
        .into_iter()
        .map(|(opaque_type_key, hidden_type)| {
            let hidden_type = infcx.resolve_vars_if_possible(hidden_type);
            register_member_constraints(
                typeck,
                &mut member_constraints,
                opaque_type_key,
                hidden_type,
            );
            trace!("finalized opaque type {:?} to {:#?}", opaque_type_key, hidden_type.ty.kind());
            if hidden_type.has_non_region_infer() {
                span_bug!(hidden_type.span, "could not resolve {:?}", hidden_type.ty);
            }

            // Convert all regions to nll vars.
            let (opaque_type_key, hidden_type) =
                fold_regions(infcx.tcx, (opaque_type_key, hidden_type), |r, _| {
                    ty::Region::new_var(infcx.tcx, typeck.to_region_vid(r))
                });

            (opaque_type_key, hidden_type)
        })
        .collect();
    assert!(typeck.constraints.member_constraints.is_empty());
    typeck.constraints.member_constraints = member_constraints;
    opaque_types
}

/// Given the map `opaque_types` containing the opaque
/// `impl Trait` types whose underlying, hidden types are being
/// inferred, this method adds constraints to the regions
/// appearing in those underlying hidden types to ensure that they
/// at least do not refer to random scopes within the current
/// function. These constraints are not (quite) sufficient to
/// guarantee that the regions are actually legal values; that
/// final condition is imposed after region inference is done.
///
/// # The Problem
///
/// Let's work through an example to explain how it works. Assume
/// the current function is as follows:
///
/// ```text
/// fn foo<'a, 'b>(..) -> (impl Bar<'a>, impl Bar<'b>)
/// ```
///
/// Here, we have two `impl Trait` types whose values are being
/// inferred (the `impl Bar<'a>` and the `impl
/// Bar<'b>`). Conceptually, this is sugar for a setup where we
/// define underlying opaque types (`Foo1`, `Foo2`) and then, in
/// the return type of `foo`, we *reference* those definitions:
///
/// ```text
/// type Foo1<'x> = impl Bar<'x>;
/// type Foo2<'x> = impl Bar<'x>;
/// fn foo<'a, 'b>(..) -> (Foo1<'a>, Foo2<'b>) { .. }
///                    //  ^^^^ ^^
///                    //  |    |
///                    //  |    args
///                    //  def_id
/// ```
///
/// As indicating in the comments above, each of those references
/// is (in the compiler) basically generic parameters (`args`)
/// applied to the type of a suitable `def_id` (which identifies
/// `Foo1` or `Foo2`).
///
/// Now, at this point in compilation, what we have done is to
/// replace each of the references (`Foo1<'a>`, `Foo2<'b>`) with
/// fresh inference variables C1 and C2. We wish to use the values
/// of these variables to infer the underlying types of `Foo1` and
/// `Foo2`. That is, this gives rise to higher-order (pattern) unification
/// constraints like:
///
/// ```text
/// for<'a> (Foo1<'a> = C1)
/// for<'b> (Foo1<'b> = C2)
/// ```
///
/// For these equation to be satisfiable, the types `C1` and `C2`
/// can only refer to a limited set of regions. For example, `C1`
/// can only refer to `'static` and `'a`, and `C2` can only refer
/// to `'static` and `'b`. The job of this function is to impose that
/// constraint.
///
/// Up to this point, C1 and C2 are basically just random type
/// inference variables, and hence they may contain arbitrary
/// regions. In fact, it is fairly likely that they do! Consider
/// this possible definition of `foo`:
///
/// ```text
/// fn foo<'a, 'b>(x: &'a i32, y: &'b i32) -> (impl Bar<'a>, impl Bar<'b>) {
///         (&*x, &*y)
///     }
/// ```
///
/// Here, the values for the concrete types of the two impl
/// traits will include inference variables:
///
/// ```text
/// &'0 i32
/// &'1 i32
/// ```
///
/// Ordinarily, the subtyping rules would ensure that these are
/// sufficiently large. But since `impl Bar<'a>` isn't a specific
/// type per se, we don't get such constraints by default. This
/// is where this function comes into play. It adds extra
/// constraints to ensure that all the regions which appear in the
/// inferred type are regions that could validly appear.
///
/// This is actually a bit of a tricky constraint in general. We
/// want to say that each variable (e.g., `'0`) can only take on
/// values that were supplied as arguments to the opaque type
/// (e.g., `'a` for `Foo1<'a>`) or `'static`, which is always in
/// scope. We don't have a constraint quite of this kind in the current
/// region checker.
///
/// # The Solution
///
/// We generally prefer to make `<=` constraints, since they
/// integrate best into the region solver. To do that, we find the
/// "minimum" of all the arguments that appear in the args: that
/// is, some region which is less than all the others. In the case
/// of `Foo1<'a>`, that would be `'a` (it's the only choice, after
/// all). Then we apply that as a least bound to the variables
/// (e.g., `'a <= '0`).
///
/// In some cases, there is no minimum. Consider this example:
///
/// ```text
/// fn baz<'a, 'b>() -> impl Trait<'a, 'b> { ... }
/// ```
///
/// Here we would report a more complex "in constraint", like `'r
/// in ['a, 'b, 'static]` (where `'r` is some region appearing in
/// the hidden type).
///
/// # Constrain regions, not the hidden concrete type
///
/// Note that generating constraints on each region `Rc` is *not*
/// the same as generating an outlives constraint on `Tc` itself.
/// For example, if we had a function like this:
///
/// ```
/// # #![feature(type_alias_impl_trait)]
/// # fn main() {}
/// # trait Foo<'a> {}
/// # impl<'a, T> Foo<'a> for (&'a u32, T) {}
/// fn foo<'a, T>(x: &'a u32, y: T) -> impl Foo<'a> {
///   (x, y)
/// }
///
/// // Equivalent to:
/// # mod dummy { use super::*;
/// type FooReturn<'a, T> = impl Foo<'a>;
/// #[define_opaque(FooReturn)]
/// fn foo<'a, T>(x: &'a u32, y: T) -> FooReturn<'a, T> {
///   (x, y)
/// }
/// # }
/// ```
///
/// then the hidden type `Tc` would be `(&'0 u32, T)` (where `'0`
/// is an inference variable). If we generated a constraint that
/// `Tc: 'a`, then this would incorrectly require that `T: 'a` --
/// but this is not necessary, because the opaque type we
/// create will be allowed to reference `T`. So we only generate a
/// constraint that `'0: 'a`.
fn register_member_constraints<'tcx>(
    typeck: &mut TypeChecker<'_, 'tcx>,
    member_constraints: &mut MemberConstraintSet<'tcx, ty::RegionVid>,
    opaque_type_key: OpaqueTypeKey<'tcx>,
    OpaqueHiddenType { span, ty: hidden_ty }: OpaqueHiddenType<'tcx>,
) {
    let tcx = typeck.tcx();
    let hidden_ty = typeck.infcx.resolve_vars_if_possible(hidden_ty);
    debug!(?hidden_ty);

    let variances = tcx.variances_of(opaque_type_key.def_id);
    debug!(?variances);

    // For a case like `impl Foo<'a, 'b>`, we would generate a constraint
    // `'r in ['a, 'b, 'static]` for each region `'r` that appears in the
    // hidden type (i.e., it must be equal to `'a`, `'b`, or `'static`).
    //
    // `conflict1` and `conflict2` are the two region bounds that we
    // detected which were unrelated. They are used for diagnostics.

    // Create the set of choice regions: each region in the hidden
    // type can be equal to any of the region parameters of the
    // opaque type definition.
    let fr_static = typeck.universal_regions.fr_static;
    let choice_regions: Vec<_> = opaque_type_key
        .args
        .iter()
        .enumerate()
        .filter(|(i, _)| variances[*i] == ty::Invariant)
        .filter_map(|(_, arg)| match arg.kind() {
            GenericArgKind::Lifetime(r) => Some(typeck.to_region_vid(r)),
            GenericArgKind::Type(_) | GenericArgKind::Const(_) => None,
        })
        .chain(iter::once(fr_static))
        .collect();

    // FIXME(#42940): This should use the `FreeRegionsVisitor`, but that's
    // not currently sound until we have existential regions.
    hidden_ty.visit_with(&mut ConstrainOpaqueTypeRegionVisitor {
        tcx,
        op: |r| {
            member_constraints.add_member_constraint(
                opaque_type_key,
                hidden_ty,
                span,
                typeck.to_region_vid(r),
                &choice_regions,
            )
        },
    });
}

/// Visitor that requires that (almost) all regions in the type visited outlive
/// `least_region`. We cannot use `push_outlives_components` because regions in
/// closure signatures are not included in their outlives components. We need to
/// ensure all regions outlive the given bound so that we don't end up with,
/// say, `ReVar` appearing in a return type and causing ICEs when other
/// functions end up with region constraints involving regions from other
/// functions.
///
/// We also cannot use `for_each_free_region` because for closures it includes
/// the regions parameters from the enclosing item.
///
/// We ignore any type parameters because impl trait values are assumed to
/// capture all the in-scope type parameters.
struct ConstrainOpaqueTypeRegionVisitor<'tcx, OP: FnMut(ty::Region<'tcx>)> {
    tcx: TyCtxt<'tcx>,
    op: OP,
}

impl<'tcx, OP> TypeVisitor<TyCtxt<'tcx>> for ConstrainOpaqueTypeRegionVisitor<'tcx, OP>
where
    OP: FnMut(ty::Region<'tcx>),
{
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        match r.kind() {
            // ignore bound regions, keep visiting
            ty::ReBound(_, _) => {}
            _ => (self.op)(r),
        }
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        // We're only interested in types involving regions
        if !ty.flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS) {
            return;
        }

        match *ty.kind() {
            ty::Closure(_, args) => {
                // Skip lifetime parameters of the enclosing item(s)

                for upvar in args.as_closure().upvar_tys() {
                    upvar.visit_with(self);
                }
                args.as_closure().sig_as_fn_ptr_ty().visit_with(self);
            }

            ty::CoroutineClosure(_, args) => {
                // Skip lifetime parameters of the enclosing item(s)

                for upvar in args.as_coroutine_closure().upvar_tys() {
                    upvar.visit_with(self);
                }

                args.as_coroutine_closure().signature_parts_ty().visit_with(self);
            }

            ty::Coroutine(_, args) => {
                // Skip lifetime parameters of the enclosing item(s)
                // Also skip the witness type, because that has no free regions.

                for upvar in args.as_coroutine().upvar_tys() {
                    upvar.visit_with(self);
                }
                args.as_coroutine().return_ty().visit_with(self);
                args.as_coroutine().yield_ty().visit_with(self);
                args.as_coroutine().resume_ty().visit_with(self);
            }

            ty::Alias(kind, ty::AliasTy { def_id, args, .. })
                if let Some(variances) = self.tcx.opt_alias_variances(kind, def_id) =>
            {
                // Skip lifetime parameters that are not captured, since they do
                // not need member constraints registered for them; we'll erase
                // them (and hopefully in the future replace them with placeholders).
                for (v, s) in std::iter::zip(variances, args.iter()) {
                    if *v != ty::Bivariant {
                        s.visit_with(self);
                    }
                }
            }

            _ => {
                ty.super_visit_with(self);
            }
        }
    }
}
