use crate::infer::InferOk;
use crate::rustc_middle::ty::subst::Subst;
use crate::traits::{self, PredicateObligation};
use hir::def_id::LocalDefId;
use rustc_data_structures::vec_map::VecMap;
use rustc_hir as hir;
use rustc_middle::ty::fold::BottomUpFolder;
use rustc_middle::ty::{self, OpaqueTypeKey, Ty, TyCtxt, TypeFoldable};
use rustc_span::Span;

use crate::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};

use super::InferCtxt;

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
    /// Moreover, it returns a `OpaqueTypeMap` that would map `?0` to
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
    pub fn instantiate_opaque_types_without_resolving_projections<T: TypeFoldable<'tcx>>(
        &self,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
        value_span: Span,
    ) -> InferOk<'tcx, T> {
        let mut instantiator =
            Instantiator { infcx: self, value_span, body_id, param_env, obligations: vec![] };
        let value = instantiator.instantiate_opaque_types_in_map(value);
        InferOk { value, obligations: instantiator.obligations }
    }
}

struct Instantiator<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    value_span: Span,
    body_id: hir::HirId,
    obligations: Vec<PredicateObligation<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
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
                    // Here, the return type of `foo` references a
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

        // Ideally, we'd get the span where *this specific `ty` came
        // from*, but right now we just use the span from the overall
        // value being folded. In simple cases like `-> impl Foo`,
        // these are the same span, but not in cases like `-> (impl
        // Foo, impl Bar)`.
        let span = self.value_span;

        let ty_var = infcx
            .next_ty_var(TypeVariableOrigin { kind: TypeVariableOriginKind::TypeInference, span });

        {
            let mut infcx = self.infcx.inner.borrow_mut();
            let key = OpaqueTypeKey { def_id, substs };
            infcx.opaque_types.insert(
                key,
                OpaqueTypeDecl {
                    opaque_type: ty,
                    definition_span: span,
                    concrete_ty: ty_var,
                    origin,
                },
            );
            infcx.opaque_types_vars.insert(ty_var, ty);
            infcx.register_obligation_for_opaque_type_queue.push((key, span));
        }

        debug!("generated new type inference var {:?}", ty_var.kind());

        let item_bounds = tcx.explicit_item_bounds(def_id);
        let bounds: Vec<_> =
            item_bounds.iter().map(|(bound, _)| bound.subst(tcx, substs)).collect();

        self.obligations.reserve(bounds.len());
        for predicate in bounds {
            // Change the predicate to refer to the type variable,
            // which will be the concrete type instead of the opaque type.
            // This also instantiates nested instances of `impl Trait`.
            let predicate = self.instantiate_opaque_types_in_map(predicate);

            let cause = traits::ObligationCause::new(span, self.body_id, traits::OpaqueType);

            // Require that the predicate holds for the concrete type.
            debug!("instantiate_opaque_types: predicate={:?}", predicate);
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
pub fn may_define_opaque_type(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    opaque_hir_id: hir::HirId,
) -> bool {
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
