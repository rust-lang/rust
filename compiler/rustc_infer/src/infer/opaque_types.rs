use super::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use super::{DefineOpaqueTypes, InferResult};
use crate::errors::OpaqueHiddenTypeDiag;
use crate::infer::{InferCtxt, InferOk};
use crate::traits::{self, PredicateObligation};
use hir::def_id::{DefId, LocalDefId};
use hir::OpaqueTyOrigin;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_middle::traits::{DefiningAnchor, ObligationCause};
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::fold::BottomUpFolder;
use rustc_middle::ty::GenericArgKind;
use rustc_middle::ty::{
    self, OpaqueHiddenType, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_span::Span;
use std::ops::ControlFlow;

mod table;

pub type OpaqueTypeMap<'tcx> = FxIndexMap<OpaqueTypeKey<'tcx>, OpaqueTypeDecl<'tcx>>;
pub use table::{OpaqueTypeStorage, OpaqueTypeTable};

/// Information about the opaque types whose values we
/// are inferring in this function (these are the `impl Trait` that
/// appear in the return type).
#[derive(Clone, Debug)]
pub struct OpaqueTypeDecl<'tcx> {
    /// The hidden types that have been inferred for this opaque type.
    /// There can be multiple, but they are all `lub`ed together at the end
    /// to obtain the canonical hidden type.
    pub hidden_type: OpaqueHiddenType<'tcx>,
}

impl<'tcx> InferCtxt<'tcx> {
    /// This is a backwards compatibility hack to prevent breaking changes from
    /// lazy TAIT around RPIT handling.
    pub fn replace_opaque_types_with_inference_vars<T: TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        value: T,
        body_id: LocalDefId,
        span: Span,
        param_env: ty::ParamEnv<'tcx>,
    ) -> InferOk<'tcx, T> {
        // We handle opaque types differently in the new solver.
        if self.next_trait_solver() {
            return InferOk { value, obligations: vec![] };
        }

        if !value.has_opaque_types() {
            return InferOk { value, obligations: vec![] };
        }

        let mut obligations = vec![];
        let replace_opaque_type = |def_id: DefId| {
            def_id.as_local().is_some_and(|def_id| self.opaque_type_origin(def_id).is_some())
        };
        let value = value.fold_with(&mut BottomUpFolder {
            tcx: self.tcx,
            lt_op: |lt| lt,
            ct_op: |ct| ct,
            ty_op: |ty| match *ty.kind() {
                ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. })
                    if replace_opaque_type(def_id) =>
                {
                    let def_span = self.tcx.def_span(def_id);
                    let span = if span.contains(def_span) { def_span } else { span };
                    let code = traits::ObligationCauseCode::OpaqueReturnType(None);
                    let cause = ObligationCause::new(span, body_id, code);
                    // FIXME(compiler-errors): We probably should add a new TypeVariableOriginKind
                    // for opaque types, and then use that kind to fix the spans for type errors
                    // that we see later on.
                    let ty_var = self.next_ty_var(TypeVariableOrigin {
                        kind: TypeVariableOriginKind::OpaqueTypeInference(def_id),
                        span,
                    });
                    obligations.extend(
                        self.handle_opaque_type(ty, ty_var, true, &cause, param_env)
                            .unwrap()
                            .obligations,
                    );
                    ty_var
                }
                _ => ty,
            },
        });
        InferOk { value, obligations }
    }

    pub fn handle_opaque_type(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        a_is_expected: bool,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> InferResult<'tcx, ()> {
        if a.references_error() || b.references_error() {
            return Ok(InferOk { value: (), obligations: vec![] });
        }
        let (a, b) = if a_is_expected { (a, b) } else { (b, a) };
        let process = |a: Ty<'tcx>, b: Ty<'tcx>, a_is_expected| match *a.kind() {
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) if def_id.is_local() => {
                let def_id = def_id.expect_local();
                match self.defining_use_anchor {
                    DefiningAnchor::Bind(_) => {
                        // Check that this is `impl Trait` type is
                        // declared by `parent_def_id` -- i.e., one whose
                        // value we are inferring. At present, this is
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
                        if self.opaque_type_origin(def_id).is_none() {
                            return None;
                        }
                    }
                    DefiningAnchor::Bubble => {}
                    DefiningAnchor::Error => return None,
                };
                if let ty::Alias(ty::Opaque, ty::AliasTy { def_id: b_def_id, .. }) = *b.kind() {
                    // We could accept this, but there are various ways to handle this situation, and we don't
                    // want to make a decision on it right now. Likely this case is so super rare anyway, that
                    // no one encounters it in practice.
                    // It does occur however in `fn fut() -> impl Future<Output = i32> { async { 42 } }`,
                    // where it is of no concern, so we only check for TAITs.
                    if let Some(OpaqueTyOrigin::TyAlias { .. }) =
                        b_def_id.as_local().and_then(|b_def_id| self.opaque_type_origin(b_def_id))
                    {
                        self.tcx.sess.emit_err(OpaqueHiddenTypeDiag {
                            span: cause.span,
                            hidden_type: self.tcx.def_span(b_def_id),
                            opaque_type: self.tcx.def_span(def_id),
                        });
                    }
                }
                Some(self.register_hidden_type(
                    OpaqueTypeKey { def_id, args },
                    cause.clone(),
                    param_env,
                    b,
                    a_is_expected,
                ))
            }
            _ => None,
        };
        if let Some(res) = process(a, b, true) {
            res
        } else if let Some(res) = process(b, a, false) {
            res
        } else {
            let (a, b) = self.resolve_vars_if_possible((a, b));
            Err(TypeError::Sorts(ExpectedFound::new(true, a, b)))
        }
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
    /// is (in the compiler) basically a substitution (`args`)
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
    #[instrument(level = "debug", skip(self))]
    pub fn register_member_constraints(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        concrete_ty: Ty<'tcx>,
        span: Span,
    ) {
        let concrete_ty = self.resolve_vars_if_possible(concrete_ty);
        debug!(?concrete_ty);

        let variances = self.tcx.variances_of(opaque_type_key.def_id);
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
        let choice_regions: Lrc<Vec<ty::Region<'tcx>>> = Lrc::new(
            opaque_type_key
                .args
                .iter()
                .enumerate()
                .filter(|(i, _)| variances[*i] == ty::Variance::Invariant)
                .filter_map(|(_, arg)| match arg.unpack() {
                    GenericArgKind::Lifetime(r) => Some(r),
                    GenericArgKind::Type(_) | GenericArgKind::Const(_) => None,
                })
                .chain(std::iter::once(self.tcx.lifetimes.re_static))
                .collect(),
        );

        concrete_ty.visit_with(&mut ConstrainOpaqueTypeRegionVisitor {
            tcx: self.tcx,
            op: |r| self.member_constraint(opaque_type_key, span, concrete_ty, r, &choice_regions),
        });
    }

    /// Returns the origin of the opaque type `def_id` if we're currently
    /// in its defining scope.
    #[instrument(skip(self), level = "trace", ret)]
    pub fn opaque_type_origin(&self, def_id: LocalDefId) -> Option<OpaqueTyOrigin> {
        let opaque_hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
        let parent_def_id = match self.defining_use_anchor {
            DefiningAnchor::Bubble | DefiningAnchor::Error => return None,
            DefiningAnchor::Bind(bind) => bind,
        };

        let origin = self.tcx.opaque_type_origin(def_id);
        let in_definition_scope = match origin {
            // Async `impl Trait`
            hir::OpaqueTyOrigin::AsyncFn(parent) => parent == parent_def_id,
            // Anonymous `impl Trait`
            hir::OpaqueTyOrigin::FnReturn(parent) => parent == parent_def_id,
            // Named `type Foo = impl Bar;`
            hir::OpaqueTyOrigin::TyAlias { in_assoc_ty } => {
                if in_assoc_ty {
                    self.tcx.opaque_types_defined_by(parent_def_id).contains(&def_id)
                } else {
                    may_define_opaque_type(self.tcx, parent_def_id, opaque_hir_id)
                }
            }
        };
        in_definition_scope.then_some(origin)
    }
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
pub struct ConstrainOpaqueTypeRegionVisitor<'tcx, OP: FnMut(ty::Region<'tcx>)> {
    pub tcx: TyCtxt<'tcx>,
    pub op: OP,
}

impl<'tcx, OP> TypeVisitor<TyCtxt<'tcx>> for ConstrainOpaqueTypeRegionVisitor<'tcx, OP>
where
    OP: FnMut(ty::Region<'tcx>),
{
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
        &mut self,
        t: &ty::Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        t.super_visit_with(self);
        ControlFlow::Continue(())
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        match *r {
            // ignore bound regions, keep visiting
            ty::ReLateBound(_, _) => ControlFlow::Continue(()),
            _ => {
                (self.op)(r);
                ControlFlow::Continue(())
            }
        }
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        // We're only interested in types involving regions
        if !ty.flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS) {
            return ControlFlow::Continue(());
        }

        match ty.kind() {
            ty::Closure(_, ref args) => {
                // Skip lifetime parameters of the enclosing item(s)

                args.as_closure().tupled_upvars_ty().visit_with(self);
                args.as_closure().sig_as_fn_ptr_ty().visit_with(self);
            }

            ty::Generator(_, ref args, _) => {
                // Skip lifetime parameters of the enclosing item(s)
                // Also skip the witness type, because that has no free regions.

                args.as_generator().tupled_upvars_ty().visit_with(self);
                args.as_generator().return_ty().visit_with(self);
                args.as_generator().yield_ty().visit_with(self);
                args.as_generator().resume_ty().visit_with(self);
            }

            ty::Alias(ty::Opaque, ty::AliasTy { def_id, ref args, .. }) => {
                // Skip lifetime parameters that are not captures.
                let variances = self.tcx.variances_of(*def_id);

                for (v, s) in std::iter::zip(variances, args.iter()) {
                    if *v != ty::Variance::Bivariant {
                        s.visit_with(self);
                    }
                }
            }

            _ => {
                ty.super_visit_with(self);
            }
        }

        ControlFlow::Continue(())
    }
}

pub enum UseKind {
    DefiningUse,
    OpaqueUse,
}

impl UseKind {
    pub fn is_defining(self) -> bool {
        match self {
            UseKind::DefiningUse => true,
            UseKind::OpaqueUse => false,
        }
    }
}

impl<'tcx> InferCtxt<'tcx> {
    #[instrument(skip(self), level = "debug")]
    fn register_hidden_type(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
        a_is_expected: bool,
    ) -> InferResult<'tcx, ()> {
        let mut obligations = Vec::new();

        self.insert_hidden_type(
            opaque_type_key,
            &cause,
            param_env,
            hidden_ty,
            a_is_expected,
            &mut obligations,
        )?;

        self.add_item_bounds_for_hidden_type(
            opaque_type_key.def_id.to_def_id(),
            opaque_type_key.args,
            cause,
            param_env,
            hidden_ty,
            &mut obligations,
        );

        Ok(InferOk { value: (), obligations })
    }

    /// Insert a hidden type into the opaque type storage, equating it
    /// with any previous entries if necessary.
    ///
    /// This **does not** add the item bounds of the opaque as nested
    /// obligations. That is only necessary when normalizing the opaque
    /// itself, not when getting the opaque type constraints from
    /// somewhere else.
    pub fn insert_hidden_type(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
        a_is_expected: bool,
        obligations: &mut Vec<PredicateObligation<'tcx>>,
    ) -> Result<(), TypeError<'tcx>> {
        // Ideally, we'd get the span where *this specific `ty` came
        // from*, but right now we just use the span from the overall
        // value being folded. In simple cases like `-> impl Foo`,
        // these are the same span, but not in cases like `-> (impl
        // Foo, impl Bar)`.
        let span = cause.span;
        if self.intercrate {
            // During intercrate we do not define opaque types but instead always
            // force ambiguity unless the hidden type is known to not implement
            // our trait.
            obligations.push(traits::Obligation::new(
                self.tcx,
                cause.clone(),
                param_env,
                ty::PredicateKind::Ambiguous,
            ))
        } else {
            let prev = self
                .inner
                .borrow_mut()
                .opaque_types()
                .register(opaque_type_key, OpaqueHiddenType { ty: hidden_ty, span });
            if let Some(prev) = prev {
                obligations.extend(
                    self.at(&cause, param_env)
                        .eq_exp(DefineOpaqueTypes::Yes, a_is_expected, prev, hidden_ty)?
                        .obligations,
                );
            }
        };

        Ok(())
    }

    pub fn add_item_bounds_for_hidden_type(
        &self,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
        obligations: &mut Vec<PredicateObligation<'tcx>>,
    ) {
        let tcx = self.tcx;
        let item_bounds = tcx.explicit_item_bounds(def_id);

        for (predicate, _) in item_bounds.iter_instantiated_copied(tcx, args) {
            let predicate = predicate.fold_with(&mut BottomUpFolder {
                tcx,
                ty_op: |ty| match *ty.kind() {
                    // We can't normalize associated types from `rustc_infer`,
                    // but we can eagerly register inference variables for them.
                    // FIXME(RPITIT): Don't replace RPITITs with inference vars.
                    // FIXME(inherent_associated_types): Extend this to support `ty::Inherent`, too.
                    ty::Alias(ty::Projection, projection_ty)
                        if !projection_ty.has_escaping_bound_vars()
                            && !tcx.is_impl_trait_in_trait(projection_ty.def_id)
                            && !self.next_trait_solver() =>
                    {
                        self.infer_projection(
                            param_env,
                            projection_ty,
                            cause.clone(),
                            0,
                            obligations,
                        )
                    }
                    // Replace all other mentions of the same opaque type with the hidden type,
                    // as the bounds must hold on the hidden type after all.
                    ty::Alias(ty::Opaque, ty::AliasTy { def_id: def_id2, args: args2, .. })
                        if def_id == def_id2 && args == args2 =>
                    {
                        hidden_ty
                    }
                    // FIXME(RPITIT): This can go away when we move to associated types
                    // FIXME(inherent_associated_types): Extend this to support `ty::Inherent`, too.
                    ty::Alias(ty::Projection, ty::AliasTy { def_id: def_id2, args: args2, .. })
                        if def_id == def_id2 && args == args2 =>
                    {
                        hidden_ty
                    }
                    _ => ty,
                },
                lt_op: |lt| lt,
                ct_op: |ct| ct,
            });

            if let ty::ClauseKind::Projection(projection) = predicate.kind().skip_binder() {
                if projection.term.references_error() {
                    // No point on adding any obligations since there's a type error involved.
                    obligations.clear();
                    return;
                }
            }
            // Require that the predicate holds for the concrete type.
            debug!(?predicate);
            obligations.push(traits::Obligation::new(
                self.tcx,
                cause.clone(),
                param_env,
                predicate,
            ));
        }
    }
}

/// Returns `true` if `opaque_hir_id` is a sibling or a child of a sibling of `def_id`.
///
/// Example:
/// ```ignore UNSOLVED (is this a bug?)
/// # #![feature(type_alias_impl_trait)]
/// pub mod foo {
///     pub mod bar {
///         pub trait Bar { /* ... */ }
///         pub type Baz = impl Bar;
///
///         # impl Bar for () {}
///         fn f1() -> Baz { /* ... */ }
///     }
///     fn f2() -> bar::Baz { /* ... */ }
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
        hir_id = tcx.hir().get_parent_item(hir_id).into();
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
