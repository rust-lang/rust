use crate::traits::{self, ObligationCause, PredicateObligation};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::infer::error_reporting::unexpected_hidden_region_diagnostic;
use rustc_infer::infer::opaque_types::OpaqueTypeDecl;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{InferCtxt, InferOk};
use rustc_middle::ty::fold::{BottomUpFolder, TypeFoldable, TypeFolder, TypeVisitor};
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, InternalSubsts, Subst};
use rustc_middle::ty::{self, OpaqueTypeKey, Ty, TyCtxt};
use rustc_span::Span;

use std::ops::ControlFlow;

pub trait InferCtxtExt<'tcx> {
    fn instantiate_opaque_types<T: TypeFoldable<'tcx>>(
        &self,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
        value_span: Span,
    ) -> InferOk<'tcx, T>;

    fn constrain_opaque_types(&self);

    fn constrain_opaque_type(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        opaque_defn: &OpaqueTypeDecl<'tcx>,
    );

    /*private*/
    fn generate_member_constraint(
        &self,
        concrete_ty: Ty<'tcx>,
        opaque_defn: &OpaqueTypeDecl<'tcx>,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        first_own_region_index: usize,
    );

    fn infer_opaque_definition_from_instantiation(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        instantiated_ty: Ty<'tcx>,
        span: Span,
    ) -> Ty<'tcx>;
}

impl<'a, 'tcx> InferCtxtExt<'tcx> for InferCtxt<'a, 'tcx> {
    /// Replaces all opaque types in `value` with fresh inference variables
    /// and creates appropriate obligations. For example, given the input:
    ///
    ///     impl Iterator<Item = impl Debug>
    ///
    /// this method would create two type variables, `?0` and `?1`. It would
    /// return the type `?0` but also the obligations:
    ///
    ///     ?0: Iterator<Item = ?1>
    ///     ?1: Debug
    ///
    /// Moreover, it returns an `OpaqueTypeMap` that would map `?0` to
    /// info about the `impl Iterator<..>` type and `?1` to info about
    /// the `impl Debug` type.
    ///
    /// # Parameters
    ///
    /// - `parent_def_id` -- the `DefId` of the function in which the opaque type
    ///   is defined
    /// - `body_id` -- the body-id with which the resulting obligations should
    ///   be associated
    /// - `param_env` -- the in-scope parameter environment to be used for
    ///   obligations
    /// - `value` -- the value within which we are instantiating opaque types
    /// - `value_span` -- the span where the value came from, used in error reporting
    fn instantiate_opaque_types<T: TypeFoldable<'tcx>>(
        &self,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
        value_span: Span,
    ) -> InferOk<'tcx, T> {
        debug!(
            "instantiate_opaque_types(value={:?}, body_id={:?}, \
             param_env={:?}, value_span={:?})",
            value, body_id, param_env, value_span,
        );
        let mut instantiator =
            Instantiator { infcx: self, body_id, param_env, value_span, obligations: vec![] };
        let value = instantiator.instantiate_opaque_types_in_map(value);
        InferOk { value, obligations: instantiator.obligations }
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
    ///                    //  |    substs
    ///                    //  def_id
    /// ```
    ///
    /// As indicating in the comments above, each of those references
    /// is (in the compiler) basically a substitution (`substs`)
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
    /// "minimum" of all the arguments that appear in the substs: that
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
    /// the same as generating an outlives constraint on `Tc` iself.
    /// For example, if we had a function like this:
    ///
    /// ```rust
    /// fn foo<'a, T>(x: &'a u32, y: T) -> impl Foo<'a> {
    ///   (x, y)
    /// }
    ///
    /// // Equivalent to:
    /// type FooReturn<'a, T> = impl Foo<'a>;
    /// fn foo<'a, T>(..) -> FooReturn<'a, T> { .. }
    /// ```
    ///
    /// then the hidden type `Tc` would be `(&'0 u32, T)` (where `'0`
    /// is an inference variable). If we generated a constraint that
    /// `Tc: 'a`, then this would incorrectly require that `T: 'a` --
    /// but this is not necessary, because the opaque type we
    /// create will be allowed to reference `T`. So we only generate a
    /// constraint that `'0: 'a`.
    ///
    /// # The `free_region_relations` parameter
    ///
    /// The `free_region_relations` argument is used to find the
    /// "minimum" of the regions supplied to a given opaque type.
    /// It must be a relation that can answer whether `'a <= 'b`,
    /// where `'a` and `'b` are regions that appear in the "substs"
    /// for the opaque type references (the `<'a>` in `Foo1<'a>`).
    ///
    /// Note that we do not impose the constraints based on the
    /// generic regions from the `Foo1` definition (e.g., `'x`). This
    /// is because the constraints we are imposing here is basically
    /// the concern of the one generating the constraining type C1,
    /// which is the current function. It also means that we can
    /// take "implied bounds" into account in some cases:
    ///
    /// ```text
    /// trait SomeTrait<'a, 'b> { }
    /// fn foo<'a, 'b>(_: &'a &'b u32) -> impl SomeTrait<'a, 'b> { .. }
    /// ```
    ///
    /// Here, the fact that `'b: 'a` is known only because of the
    /// implied bounds from the `&'a &'b u32` parameter, and is not
    /// "inherent" to the opaque type definition.
    ///
    /// # Parameters
    ///
    /// - `opaque_types` -- the map produced by `instantiate_opaque_types`
    /// - `free_region_relations` -- something that can be used to relate
    ///   the free regions (`'a`) that appear in the impl trait.
    fn constrain_opaque_types(&self) {
        let opaque_types = self.inner.borrow().opaque_types.clone();
        for (opaque_type_key, opaque_defn) in opaque_types {
            self.constrain_opaque_type(opaque_type_key, &opaque_defn);
        }
    }

    /// See `constrain_opaque_types` for documentation.
    #[instrument(level = "debug", skip(self))]
    fn constrain_opaque_type(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        opaque_defn: &OpaqueTypeDecl<'tcx>,
    ) {
        let def_id = opaque_type_key.def_id;

        let tcx = self.tcx;

        let concrete_ty = self.resolve_vars_if_possible(opaque_defn.concrete_ty);

        debug!(?concrete_ty);

        let first_own_region = match opaque_defn.origin {
            hir::OpaqueTyOrigin::FnReturn | hir::OpaqueTyOrigin::AsyncFn => {
                // We lower
                //
                // fn foo<'l0..'ln>() -> impl Trait<'l0..'lm>
                //
                // into
                //
                // type foo::<'p0..'pn>::Foo<'q0..'qm>
                // fn foo<l0..'ln>() -> foo::<'static..'static>::Foo<'l0..'lm>.
                //
                // For these types we only iterate over `'l0..lm` below.
                tcx.generics_of(def_id).parent_count
            }
            // These opaque type inherit all lifetime parameters from their
            // parent, so we have to check them all.
            hir::OpaqueTyOrigin::TyAlias => 0,
        };

        // The regions that appear in the hidden type must be equal to
        // one of the regions in scope for the opaque type.
        self.generate_member_constraint(
            concrete_ty,
            opaque_defn,
            opaque_type_key,
            first_own_region,
        );
    }

    /// As a fallback, we sometimes generate an "in constraint". For
    /// a case like `impl Foo<'a, 'b>`, where `'a` and `'b` cannot be
    /// related, we would generate a constraint `'r in ['a, 'b,
    /// 'static]` for each region `'r` that appears in the hidden type
    /// (i.e., it must be equal to `'a`, `'b`, or `'static`).
    ///
    /// `conflict1` and `conflict2` are the two region bounds that we
    /// detected which were unrelated. They are used for diagnostics.
    fn generate_member_constraint(
        &self,
        concrete_ty: Ty<'tcx>,
        opaque_defn: &OpaqueTypeDecl<'tcx>,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        first_own_region: usize,
    ) {
        // Create the set of choice regions: each region in the hidden
        // type can be equal to any of the region parameters of the
        // opaque type definition.
        let choice_regions: Lrc<Vec<ty::Region<'tcx>>> = Lrc::new(
            opaque_type_key.substs[first_own_region..]
                .iter()
                .filter_map(|arg| match arg.unpack() {
                    GenericArgKind::Lifetime(r) => Some(r),
                    GenericArgKind::Type(_) | GenericArgKind::Const(_) => None,
                })
                .chain(std::iter::once(self.tcx.lifetimes.re_static))
                .collect(),
        );

        concrete_ty.visit_with(&mut ConstrainOpaqueTypeRegionVisitor {
            tcx: self.tcx,
            op: |r| {
                self.member_constraint(
                    opaque_type_key.def_id,
                    opaque_defn.definition_span,
                    concrete_ty,
                    r,
                    &choice_regions,
                )
            },
        });
    }

    /// Given the fully resolved, instantiated type for an opaque
    /// type, i.e., the value of an inference variable like C1 or C2
    /// (*), computes the "definition type" for an opaque type
    /// definition -- that is, the inferred value of `Foo1<'x>` or
    /// `Foo2<'x>` that we would conceptually use in its definition:
    ///
    ///     type Foo1<'x> = impl Bar<'x> = AAA; <-- this type AAA
    ///     type Foo2<'x> = impl Bar<'x> = BBB; <-- or this type BBB
    ///     fn foo<'a, 'b>(..) -> (Foo1<'a>, Foo2<'b>) { .. }
    ///
    /// Note that these values are defined in terms of a distinct set of
    /// generic parameters (`'x` instead of `'a`) from C1 or C2. The main
    /// purpose of this function is to do that translation.
    ///
    /// (*) C1 and C2 were introduced in the comments on
    /// `constrain_opaque_types`. Read that comment for more context.
    ///
    /// # Parameters
    ///
    /// - `def_id`, the `impl Trait` type
    /// - `substs`, the substs  used to instantiate this opaque type
    /// - `instantiated_ty`, the inferred type C1 -- fully resolved, lifted version of
    ///   `opaque_defn.concrete_ty`
    #[instrument(level = "debug", skip(self))]
    fn infer_opaque_definition_from_instantiation(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        instantiated_ty: Ty<'tcx>,
        span: Span,
    ) -> Ty<'tcx> {
        let OpaqueTypeKey { def_id, substs } = opaque_type_key;

        // Use substs to build up a reverse map from regions to their
        // identity mappings. This is necessary because of `impl
        // Trait` lifetimes are computed by replacing existing
        // lifetimes with 'static and remapping only those used in the
        // `impl Trait` return type, resulting in the parameters
        // shifting.
        let id_substs = InternalSubsts::identity_for_item(self.tcx, def_id);
        debug!(?id_substs);
        let map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>> =
            substs.iter().enumerate().map(|(index, subst)| (subst, id_substs[index])).collect();
        debug!("map = {:#?}", map);

        // Convert the type from the function into a type valid outside
        // the function, by replacing invalid regions with 'static,
        // after producing an error for each of them.
        let definition_ty = instantiated_ty.fold_with(&mut ReverseMapper::new(
            self.tcx,
            self.is_tainted_by_errors(),
            def_id,
            map,
            instantiated_ty,
            span,
        ));
        debug!(?definition_ty);

        definition_ty
    }
}

// Visitor that requires that (almost) all regions in the type visited outlive
// `least_region`. We cannot use `push_outlives_components` because regions in
// closure signatures are not included in their outlives components. We need to
// ensure all regions outlive the given bound so that we don't end up with,
// say, `ReVar` appearing in a return type and causing ICEs when other
// functions end up with region constraints involving regions from other
// functions.
//
// We also cannot use `for_each_free_region` because for closures it includes
// the regions parameters from the enclosing item.
//
// We ignore any type parameters because impl trait values are assumed to
// capture all the in-scope type parameters.
struct ConstrainOpaqueTypeRegionVisitor<'tcx, OP> {
    tcx: TyCtxt<'tcx>,
    op: OP,
}

impl<'tcx, OP> TypeVisitor<'tcx> for ConstrainOpaqueTypeRegionVisitor<'tcx, OP>
where
    OP: FnMut(ty::Region<'tcx>),
{
    fn tcx_for_anon_const_substs(&self) -> Option<TyCtxt<'tcx>> {
        Some(self.tcx)
    }

    fn visit_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: &ty::Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        t.as_ref().skip_binder().visit_with(self);
        ControlFlow::CONTINUE
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        match *r {
            // ignore bound regions, keep visiting
            ty::ReLateBound(_, _) => ControlFlow::CONTINUE,
            _ => {
                (self.op)(r);
                ControlFlow::CONTINUE
            }
        }
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        // We're only interested in types involving regions
        if !ty.flags().intersects(ty::TypeFlags::HAS_POTENTIAL_FREE_REGIONS) {
            return ControlFlow::CONTINUE;
        }

        match ty.kind() {
            ty::Closure(_, ref substs) => {
                // Skip lifetime parameters of the enclosing item(s)

                substs.as_closure().tupled_upvars_ty().visit_with(self);
                substs.as_closure().sig_as_fn_ptr_ty().visit_with(self);
            }

            ty::Generator(_, ref substs, _) => {
                // Skip lifetime parameters of the enclosing item(s)
                // Also skip the witness type, because that has no free regions.

                substs.as_generator().tupled_upvars_ty().visit_with(self);
                substs.as_generator().return_ty().visit_with(self);
                substs.as_generator().yield_ty().visit_with(self);
                substs.as_generator().resume_ty().visit_with(self);
            }
            _ => {
                ty.super_visit_with(self);
            }
        }

        ControlFlow::CONTINUE
    }
}

struct ReverseMapper<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// If errors have already been reported in this fn, we suppress
    /// our own errors because they are sometimes derivative.
    tainted_by_errors: bool,

    opaque_type_def_id: DefId,
    map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>>,
    map_missing_regions_to_empty: bool,

    /// initially `Some`, set to `None` once error has been reported
    hidden_ty: Option<Ty<'tcx>>,

    /// Span of function being checked.
    span: Span,
}

impl ReverseMapper<'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        tainted_by_errors: bool,
        opaque_type_def_id: DefId,
        map: FxHashMap<GenericArg<'tcx>, GenericArg<'tcx>>,
        hidden_ty: Ty<'tcx>,
        span: Span,
    ) -> Self {
        Self {
            tcx,
            tainted_by_errors,
            opaque_type_def_id,
            map,
            map_missing_regions_to_empty: false,
            hidden_ty: Some(hidden_ty),
            span,
        }
    }

    fn fold_kind_mapping_missing_regions_to_empty(
        &mut self,
        kind: GenericArg<'tcx>,
    ) -> GenericArg<'tcx> {
        assert!(!self.map_missing_regions_to_empty);
        self.map_missing_regions_to_empty = true;
        let kind = kind.fold_with(self);
        self.map_missing_regions_to_empty = false;
        kind
    }

    fn fold_kind_normally(&mut self, kind: GenericArg<'tcx>) -> GenericArg<'tcx> {
        assert!(!self.map_missing_regions_to_empty);
        kind.fold_with(self)
    }
}

impl TypeFolder<'tcx> for ReverseMapper<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[instrument(skip(self), level = "debug")]
    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r {
            // Ignore bound regions and `'static` regions that appear in the
            // type, we only need to remap regions that reference lifetimes
            // from the function declaraion.
            // This would ignore `'r` in a type like `for<'r> fn(&'r u32)`.
            ty::ReLateBound(..) | ty::ReStatic => return r,

            // If regions have been erased (by writeback), don't try to unerase
            // them.
            ty::ReErased => return r,

            // The regions that we expect from borrow checking.
            ty::ReEarlyBound(_) | ty::ReFree(_) | ty::ReEmpty(ty::UniverseIndex::ROOT) => {}

            ty::ReEmpty(_) | ty::RePlaceholder(_) | ty::ReVar(_) => {
                // All of the regions in the type should either have been
                // erased by writeback, or mapped back to named regions by
                // borrow checking.
                bug!("unexpected region kind in opaque type: {:?}", r);
            }
        }

        let generics = self.tcx().generics_of(self.opaque_type_def_id);
        match self.map.get(&r.into()).map(|k| k.unpack()) {
            Some(GenericArgKind::Lifetime(r1)) => r1,
            Some(u) => panic!("region mapped to unexpected kind: {:?}", u),
            None if self.map_missing_regions_to_empty || self.tainted_by_errors => {
                self.tcx.lifetimes.re_root_empty
            }
            None if generics.parent.is_some() => {
                if let Some(hidden_ty) = self.hidden_ty.take() {
                    unexpected_hidden_region_diagnostic(
                        self.tcx,
                        self.tcx.def_span(self.opaque_type_def_id),
                        hidden_ty,
                        r,
                    )
                    .emit();
                }
                self.tcx.lifetimes.re_root_empty
            }
            None => {
                self.tcx
                    .sess
                    .struct_span_err(self.span, "non-defining opaque type use in defining scope")
                    .span_label(
                        self.span,
                        format!(
                            "lifetime `{}` is part of concrete type but not used in \
                                 parameter list of the `impl Trait` type alias",
                            r
                        ),
                    )
                    .emit();

                self.tcx().lifetimes.re_static
            }
        }
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Closure(def_id, substs) => {
                // I am a horrible monster and I pray for death. When
                // we encounter a closure here, it is always a closure
                // from within the function that we are currently
                // type-checking -- one that is now being encapsulated
                // in an opaque type. Ideally, we would
                // go through the types/lifetimes that it references
                // and treat them just like we would any other type,
                // which means we would error out if we find any
                // reference to a type/region that is not in the
                // "reverse map".
                //
                // **However,** in the case of closures, there is a
                // somewhat subtle (read: hacky) consideration. The
                // problem is that our closure types currently include
                // all the lifetime parameters declared on the
                // enclosing function, even if they are unused by the
                // closure itself. We can't readily filter them out,
                // so here we replace those values with `'empty`. This
                // can't really make a difference to the rest of the
                // compiler; those regions are ignored for the
                // outlives relation, and hence don't affect trait
                // selection or auto traits, and they are erased
                // during codegen.

                let generics = self.tcx.generics_of(def_id);
                let substs = self.tcx.mk_substs(substs.iter().enumerate().map(|(index, kind)| {
                    if index < generics.parent_count {
                        // Accommodate missing regions in the parent kinds...
                        self.fold_kind_mapping_missing_regions_to_empty(kind)
                    } else {
                        // ...but not elsewhere.
                        self.fold_kind_normally(kind)
                    }
                }));

                self.tcx.mk_closure(def_id, substs)
            }

            ty::Generator(def_id, substs, movability) => {
                let generics = self.tcx.generics_of(def_id);
                let substs = self.tcx.mk_substs(substs.iter().enumerate().map(|(index, kind)| {
                    if index < generics.parent_count {
                        // Accommodate missing regions in the parent kinds...
                        self.fold_kind_mapping_missing_regions_to_empty(kind)
                    } else {
                        // ...but not elsewhere.
                        self.fold_kind_normally(kind)
                    }
                }));

                self.tcx.mk_generator(def_id, substs, movability)
            }

            ty::Param(param) => {
                // Look it up in the substitution list.
                match self.map.get(&ty.into()).map(|k| k.unpack()) {
                    // Found it in the substitution list; replace with the parameter from the
                    // opaque type.
                    Some(GenericArgKind::Type(t1)) => t1,
                    Some(u) => panic!("type mapped to unexpected kind: {:?}", u),
                    None => {
                        debug!(?param, ?self.map);
                        self.tcx
                            .sess
                            .struct_span_err(
                                self.span,
                                &format!(
                                    "type parameter `{}` is part of concrete type but not \
                                          used in parameter list for the `impl Trait` type alias",
                                    ty
                                ),
                            )
                            .emit();

                        self.tcx().ty_error()
                    }
                }
            }

            _ => ty.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        trace!("checking const {:?}", ct);
        // Find a const parameter
        match ct.val {
            ty::ConstKind::Param(..) => {
                // Look it up in the substitution list.
                match self.map.get(&ct.into()).map(|k| k.unpack()) {
                    // Found it in the substitution list, replace with the parameter from the
                    // opaque type.
                    Some(GenericArgKind::Const(c1)) => c1,
                    Some(u) => panic!("const mapped to unexpected kind: {:?}", u),
                    None => {
                        self.tcx
                            .sess
                            .struct_span_err(
                                self.span,
                                &format!(
                                    "const parameter `{}` is part of concrete type but not \
                                          used in parameter list for the `impl Trait` type alias",
                                    ct
                                ),
                            )
                            .emit();

                        self.tcx().const_error(ct.ty)
                    }
                }
            }

            _ => ct,
        }
    }
}

struct Instantiator<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    body_id: hir::HirId,
    param_env: ty::ParamEnv<'tcx>,
    value_span: Span,
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'a, 'tcx> Instantiator<'a, 'tcx> {
    fn instantiate_opaque_types_in_map<T: TypeFoldable<'tcx>>(&mut self, value: T) -> T {
        let tcx = self.infcx.tcx;
        value.fold_with(&mut BottomUpFolder {
            tcx,
            ty_op: |ty| {
                if ty.references_error() {
                    return tcx.ty_error();
                } else if let ty::Opaque(def_id, substs) = ty.kind() {
                    // Check that this is `impl Trait` type is
                    // declared by `parent_def_id` -- i.e., one whose
                    // value we are inferring.  At present, this is
                    // always true during the first phase of
                    // type-check, but not always true later on during
                    // NLL. Once we support named opaque types more fully,
                    // this same scenario will be able to arise during all phases.
                    //
                    // Here is an example using type alias `impl Trait`
                    // that indicates the distinction we are checking for:
                    //
                    // ```rust
                    // mod a {
                    //   pub type Foo = impl Iterator;
                    //   pub fn make_foo() -> Foo { .. }
                    // }
                    //
                    // mod b {
                    //   fn foo() -> a::Foo { a::make_foo() }
                    // }
                    // ```
                    //
                    // Here, the return type of `foo` references an
                    // `Opaque` indeed, but not one whose value is
                    // presently being inferred. You can get into a
                    // similar situation with closure return types
                    // today:
                    //
                    // ```rust
                    // fn foo() -> impl Iterator { .. }
                    // fn bar() {
                    //     let x = || foo(); // returns the Opaque assoc with `foo`
                    // }
                    // ```
                    if let Some(def_id) = def_id.as_local() {
                        let opaque_hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
                        let parent_def_id = self.infcx.defining_use_anchor;
                        let def_scope_default = || {
                            let opaque_parent_hir_id = tcx.hir().get_parent_item(opaque_hir_id);
                            parent_def_id == tcx.hir().local_def_id(opaque_parent_hir_id)
                        };
                        let (in_definition_scope, origin) =
                            match tcx.hir().expect_item(opaque_hir_id).kind {
                                // Anonymous `impl Trait`
                                hir::ItemKind::OpaqueTy(hir::OpaqueTy {
                                    impl_trait_fn: Some(parent),
                                    origin,
                                    ..
                                }) => (parent == parent_def_id.to_def_id(), origin),
                                // Named `type Foo = impl Bar;`
                                hir::ItemKind::OpaqueTy(hir::OpaqueTy {
                                    impl_trait_fn: None,
                                    origin,
                                    ..
                                }) => (
                                    may_define_opaque_type(tcx, parent_def_id, opaque_hir_id),
                                    origin,
                                ),
                                _ => (def_scope_default(), hir::OpaqueTyOrigin::TyAlias),
                            };
                        if in_definition_scope {
                            let opaque_type_key =
                                OpaqueTypeKey { def_id: def_id.to_def_id(), substs };
                            return self.fold_opaque_ty(ty, opaque_type_key, origin);
                        }

                        debug!(
                            "instantiate_opaque_types_in_map: \
                             encountered opaque outside its definition scope \
                             def_id={:?}",
                            def_id,
                        );
                    }
                }

                ty
            },
            lt_op: |lt| lt,
            ct_op: |ct| ct,
        })
    }

    #[instrument(skip(self), level = "debug")]
    fn fold_opaque_ty(
        &mut self,
        ty: Ty<'tcx>,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        origin: hir::OpaqueTyOrigin,
    ) -> Ty<'tcx> {
        let infcx = self.infcx;
        let tcx = infcx.tcx;
        let OpaqueTypeKey { def_id, substs } = opaque_type_key;

        // Use the same type variable if the exact same opaque type appears more
        // than once in the return type (e.g., if it's passed to a type alias).
        if let Some(opaque_defn) = infcx.inner.borrow().opaque_types.get(&opaque_type_key) {
            debug!("re-using cached concrete type {:?}", opaque_defn.concrete_ty.kind());
            return opaque_defn.concrete_ty;
        }

        let ty_var = infcx.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::TypeInference,
            span: self.value_span,
        });

        // Ideally, we'd get the span where *this specific `ty` came
        // from*, but right now we just use the span from the overall
        // value being folded. In simple cases like `-> impl Foo`,
        // these are the same span, but not in cases like `-> (impl
        // Foo, impl Bar)`.
        let definition_span = self.value_span;

        {
            let mut infcx = self.infcx.inner.borrow_mut();
            infcx.opaque_types.insert(
                OpaqueTypeKey { def_id, substs },
                OpaqueTypeDecl { opaque_type: ty, definition_span, concrete_ty: ty_var, origin },
            );
            infcx.opaque_types_vars.insert(ty_var, ty);
        }

        debug!("generated new type inference var {:?}", ty_var.kind());

        let item_bounds = tcx.explicit_item_bounds(def_id);

        self.obligations.reserve(item_bounds.len());
        for (predicate, _) in item_bounds {
            debug!(?predicate);
            let predicate = predicate.subst(tcx, substs);
            debug!(?predicate);

            // We can't normalize associated types from `rustc_infer`, but we can eagerly register inference variables for them.
            let predicate = predicate.fold_with(&mut BottomUpFolder {
                tcx,
                ty_op: |ty| match ty.kind() {
                    ty::Projection(projection_ty) => infcx.infer_projection(
                        self.param_env,
                        *projection_ty,
                        ObligationCause::misc(self.value_span, self.body_id),
                        0,
                        &mut self.obligations,
                    ),
                    _ => ty,
                },
                lt_op: |lt| lt,
                ct_op: |ct| ct,
            });
            debug!(?predicate);

            if let ty::PredicateKind::Projection(projection) = predicate.kind().skip_binder() {
                if projection.ty.references_error() {
                    // No point on adding these obligations since there's a type error involved.
                    return tcx.ty_error();
                }
            }
            // Change the predicate to refer to the type variable,
            // which will be the concrete type instead of the opaque type.
            // This also instantiates nested instances of `impl Trait`.
            let predicate = self.instantiate_opaque_types_in_map(predicate);

            let cause =
                traits::ObligationCause::new(self.value_span, self.body_id, traits::OpaqueType);

            // Require that the predicate holds for the concrete type.
            debug!(?predicate);
            self.obligations.push(traits::Obligation::new(cause, self.param_env, predicate));
        }

        ty_var
    }
}

/// Returns `true` if `opaque_hir_id` is a sibling or a child of a sibling of `def_id`.
///
/// Example:
/// ```rust
/// pub mod foo {
///     pub mod bar {
///         pub trait Bar { .. }
///
///         pub type Baz = impl Bar;
///
///         fn f1() -> Baz { .. }
///     }
///
///     fn f2() -> bar::Baz { .. }
/// }
/// ```
///
/// Here, `def_id` is the `LocalDefId` of the defining use of the opaque type (e.g., `f1` or `f2`),
/// and `opaque_hir_id` is the `HirId` of the definition of the opaque type `Baz`.
/// For the above example, this function returns `true` for `f1` and `false` for `f2`.
fn may_define_opaque_type(tcx: TyCtxt<'_>, def_id: LocalDefId, opaque_hir_id: hir::HirId) -> bool {
    let mut hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    // Named opaque types can be defined by any siblings or children of siblings.
    let scope = tcx.hir().get_defining_scope(opaque_hir_id);
    // We walk up the node tree until we hit the root or the scope of the opaque type.
    while hir_id != scope && hir_id != hir::CRATE_HIR_ID {
        hir_id = tcx.hir().get_parent_item(hir_id);
    }
    // Syntactically, we are allowed to define the concrete type if:
    let res = hir_id == scope;
    trace!(
        "may_define_opaque_type(def={:?}, opaque_node={:?}) = {}",
        tcx.hir().find(hir_id),
        tcx.hir().get(opaque_hir_id),
        res
    );
    res
}

/// Given a set of predicates that apply to an object type, returns
/// the region bounds that the (erased) `Self` type must
/// outlive. Precisely *because* the `Self` type is erased, the
/// parameter `erased_self_ty` must be supplied to indicate what type
/// has been used to represent `Self` in the predicates
/// themselves. This should really be a unique type; `FreshTy(0)` is a
/// popular choice.
///
/// N.B., in some cases, particularly around higher-ranked bounds,
/// this function returns a kind of conservative approximation.
/// That is, all regions returned by this function are definitely
/// required, but there may be other region bounds that are not
/// returned, as well as requirements like `for<'a> T: 'a`.
///
/// Requires that trait definitions have been processed so that we can
/// elaborate predicates and walk supertraits.
#[instrument(skip(tcx, predicates), level = "debug")]
crate fn required_region_bounds(
    tcx: TyCtxt<'tcx>,
    erased_self_ty: Ty<'tcx>,
    predicates: impl Iterator<Item = ty::Predicate<'tcx>>,
) -> Vec<ty::Region<'tcx>> {
    assert!(!erased_self_ty.has_escaping_bound_vars());

    traits::elaborate_predicates(tcx, predicates)
        .filter_map(|obligation| {
            debug!(?obligation);
            match obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Projection(..)
                | ty::PredicateKind::Trait(..)
                | ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::WellFormed(..)
                | ty::PredicateKind::ObjectSafe(..)
                | ty::PredicateKind::ClosureKind(..)
                | ty::PredicateKind::RegionOutlives(..)
                | ty::PredicateKind::ConstEvaluatable(..)
                | ty::PredicateKind::ConstEquate(..)
                | ty::PredicateKind::TypeWellFormedFromEnv(..) => None,
                ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(ref t, ref r)) => {
                    // Search for a bound of the form `erased_self_ty
                    // : 'a`, but be wary of something like `for<'a>
                    // erased_self_ty : 'a` (we interpret a
                    // higher-ranked bound like that as 'static,
                    // though at present the code in `fulfill.rs`
                    // considers such bounds to be unsatisfiable, so
                    // it's kind of a moot point since you could never
                    // construct such an object, but this seems
                    // correct even if that code changes).
                    if t == &erased_self_ty && !r.has_escaping_bound_vars() {
                        Some(*r)
                    } else {
                        None
                    }
                }
            }
        })
        .collect()
}
