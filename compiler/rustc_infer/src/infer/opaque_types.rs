use crate::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use crate::infer::{InferCtxt, InferOk};
use crate::traits;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::vec_map::VecMap;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::fold::BottomUpFolder;
use rustc_middle::ty::subst::{GenericArgKind, Subst};
use rustc_middle::ty::{self, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable, TypeVisitor};
use rustc_span::Span;

use std::ops::ControlFlow;

pub type OpaqueTypeMap<'tcx> = VecMap<OpaqueTypeKey<'tcx>, OpaqueTypeDecl<'tcx>>;

/// Information about the opaque types whose values we
/// are inferring in this function (these are the `impl Trait` that
/// appear in the return type).
#[derive(Copy, Clone, Debug)]
pub struct OpaqueTypeDecl<'tcx> {
    /// The opaque type (`ty::Opaque`) for this declaration.
    pub opaque_type: Ty<'tcx>,

    /// The span of this particular definition of the opaque type. So
    /// for example:
    ///
    /// ```ignore (incomplete snippet)
    /// type Foo = impl Baz;
    /// fn bar() -> Foo {
    /// //          ^^^ This is the span we are looking for!
    /// }
    /// ```
    ///
    /// In cases where the fn returns `(impl Trait, impl Trait)` or
    /// other such combinations, the result is currently
    /// over-approximated, but better than nothing.
    pub definition_span: Span,

    /// The type variable that represents the value of the opaque type
    /// that we require. In other words, after we compile this function,
    /// we will be created a constraint like:
    ///
    ///     Foo<'a, T> = ?C
    ///
    /// where `?C` is the value of this type variable. =) It may
    /// naturally refer to the type and lifetime parameters in scope
    /// in this function, though ultimately it should only reference
    /// those that are arguments to `Foo` in the constraint above. (In
    /// other words, `?C` should not include `'b`, even though it's a
    /// lifetime parameter on `foo`.)
    pub concrete_ty: Ty<'tcx>,

    /// The origin of the opaque type.
    pub origin: hir::OpaqueTyOrigin,
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
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
    pub fn instantiate_opaque_types<T: TypeFoldable<'tcx>>(
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
    #[instrument(level = "debug", skip(self))]
    pub fn constrain_opaque_type(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        opaque_defn: &OpaqueTypeDecl<'tcx>,
    ) {
        let def_id = opaque_type_key.def_id;

        let tcx = self.tcx;

        let concrete_ty = self.resolve_vars_if_possible(opaque_defn.concrete_ty);

        debug!(?concrete_ty);

        let first_own_region = match opaque_defn.origin {
            hir::OpaqueTyOrigin::FnReturn(..) | hir::OpaqueTyOrigin::AsyncFn(..) => {
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

        // For a case like `impl Foo<'a, 'b>`, we would generate a constraint
        // `'r in ['a, 'b, 'static]` for each region `'r` that appears in the
        // hidden type (i.e., it must be equal to `'a`, `'b`, or `'static`).
        //
        // `conflict1` and `conflict2` are the two region bounds that we
        // detected which were unrelated. They are used for diagnostics.

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

    fn opaque_type_origin(&self, def_id: LocalDefId) -> Option<hir::OpaqueTyOrigin> {
        let tcx = self.tcx;
        let opaque_hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
        let parent_def_id = self.defining_use_anchor?;
        let item_kind = &tcx.hir().expect_item(def_id).kind;
        let hir::ItemKind::OpaqueTy(hir::OpaqueTy { origin, ..  }) = item_kind else {
            span_bug!(
                tcx.def_span(def_id),
                "weird opaque type: {:#?}",
                item_kind
            )
        };
        let in_definition_scope = match *origin {
            // Async `impl Trait`
            hir::OpaqueTyOrigin::AsyncFn(parent) => parent == parent_def_id,
            // Anonymous `impl Trait`
            hir::OpaqueTyOrigin::FnReturn(parent) => parent == parent_def_id,
            // Named `type Foo = impl Bar;`
            hir::OpaqueTyOrigin::TyAlias => {
                may_define_opaque_type(tcx, parent_def_id, opaque_hir_id)
            }
        };
        in_definition_scope.then_some(*origin)
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
struct ConstrainOpaqueTypeRegionVisitor<OP> {
    op: OP,
}

impl<'tcx, OP> TypeVisitor<'tcx> for ConstrainOpaqueTypeRegionVisitor<OP>
where
    OP: FnMut(ty::Region<'tcx>),
{
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
        if !ty.flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS) {
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

struct Instantiator<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    body_id: hir::HirId,
    param_env: ty::ParamEnv<'tcx>,
    value_span: Span,
    obligations: Vec<traits::PredicateObligation<'tcx>>,
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
                        if let Some(origin) = self.infcx.opaque_type_origin(def_id) {
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

            let predicate = predicate.fold_with(&mut BottomUpFolder {
                tcx,
                ty_op: |ty| match *ty.kind() {
                    // Replace all other mentions of the same opaque type with the hidden type,
                    // as the bounds must hold on the hidden type after all.
                    ty::Opaque(def_id2, substs2) if def_id == def_id2 && substs == substs2 => {
                        ty_var
                    }
                    // Instantiate nested instances of `impl Trait`.
                    ty::Opaque(..) => self.instantiate_opaque_types_in_map(ty),
                    _ => ty,
                },
                lt_op: |lt| lt,
                ct_op: |ct| ct,
            });

            // We can't normalize associated types from `rustc_infer`, but we can eagerly register inference variables for them.
            let predicate = predicate.fold_with(&mut BottomUpFolder {
                tcx,
                ty_op: |ty| match ty.kind() {
                    ty::Projection(projection_ty) if !projection_ty.has_escaping_bound_vars() => {
                        infcx.infer_projection(
                            self.param_env,
                            *projection_ty,
                            traits::ObligationCause::misc(self.value_span, self.body_id),
                            0,
                            &mut self.obligations,
                        )
                    }
                    _ => ty,
                },
                lt_op: |lt| lt,
                ct_op: |ct| ct,
            });
            debug!(?predicate);

            if let ty::PredicateKind::Projection(projection) = predicate.kind().skip_binder() {
                if projection.term.references_error() {
                    return tcx.ty_error();
                }
            }

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
        hir_id = tcx.hir().local_def_id_to_hir_id(tcx.hir().get_parent_item(hir_id));
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
