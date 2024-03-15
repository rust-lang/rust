//! Code which is used by built-in goals that match "structurally", such a auto
//! traits, `Copy`/`Clone`.
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::LangItem;
use rustc_hir::{def_id::DefId, Movability, Mutability};
use rustc_infer::traits::query::NoSolution;
use rustc_middle::traits::solve::Goal;
use rustc_middle::ty::{
    self, ToPredicate, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};
use rustc_span::sym;

use crate::solve::EvalCtxt;

// Calculates the constituent types of a type for `auto trait` purposes.
//
// For types with an "existential" binder, i.e. coroutine witnesses, we also
// instantiate the binder with placeholders eagerly.
#[instrument(level = "debug", skip(ecx), ret)]
pub(in crate::solve) fn instantiate_constituent_tys_for_auto_trait<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<ty::Binder<'tcx, Ty<'tcx>>>, NoSolution> {
    let tcx = ecx.tcx();
    match *ty.kind() {
        ty::Uint(_)
        | ty::Int(_)
        | ty::Bool
        | ty::Float(_)
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::Error(_)
        | ty::Never
        | ty::Char => Ok(vec![]),

        // Treat `str` like it's defined as `struct str([u8]);`
        ty::Str => Ok(vec![ty::Binder::dummy(Ty::new_slice(tcx, tcx.types.u8))]),

        ty::Dynamic(..)
        | ty::Param(..)
        | ty::Foreign(..)
        | ty::Alias(ty::Projection | ty::Inherent | ty::Weak, ..)
        | ty::Placeholder(..)
        | ty::Bound(..)
        | ty::Infer(_) => {
            bug!("unexpected type `{ty}`")
        }

        ty::RawPtr(ty::TypeAndMut { ty: element_ty, .. }) | ty::Ref(_, element_ty, _) => {
            Ok(vec![ty::Binder::dummy(element_ty)])
        }

        ty::Array(element_ty, _) | ty::Slice(element_ty) => Ok(vec![ty::Binder::dummy(element_ty)]),

        ty::Tuple(tys) => {
            // (T1, ..., Tn) -- meets any bound that all of T1...Tn meet
            Ok(tys.iter().map(ty::Binder::dummy).collect())
        }

        ty::Closure(_, args) => Ok(vec![ty::Binder::dummy(args.as_closure().tupled_upvars_ty())]),

        ty::CoroutineClosure(_, args) => {
            Ok(vec![ty::Binder::dummy(args.as_coroutine_closure().tupled_upvars_ty())])
        }

        ty::Coroutine(_, args) => {
            let coroutine_args = args.as_coroutine();
            Ok(vec![
                ty::Binder::dummy(coroutine_args.tupled_upvars_ty()),
                ty::Binder::dummy(coroutine_args.witness()),
            ])
        }

        ty::CoroutineWitness(def_id, args) => Ok(ecx
            .tcx()
            .coroutine_hidden_types(def_id)
            .map(|bty| replace_erased_lifetimes_with_bound_vars(tcx, bty.instantiate(tcx, args)))
            .collect()),

        // For `PhantomData<T>`, we pass `T`.
        ty::Adt(def, args) if def.is_phantom_data() => Ok(vec![ty::Binder::dummy(args.type_at(0))]),

        ty::Adt(def, args) => {
            Ok(def.all_fields().map(|f| ty::Binder::dummy(f.ty(tcx, args))).collect())
        }

        ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) => {
            // We can resolve the `impl Trait` to its concrete type,
            // which enforces a DAG between the functions requiring
            // the auto trait bounds in question.
            Ok(vec![ty::Binder::dummy(tcx.type_of(def_id).instantiate(tcx, args))])
        }
    }
}

pub(in crate::solve) fn replace_erased_lifetimes_with_bound_vars<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> ty::Binder<'tcx, Ty<'tcx>> {
    debug_assert!(!ty.has_bound_regions());
    let mut counter = 0;
    let ty = tcx.fold_regions(ty, |r, current_depth| match r.kind() {
        ty::ReErased => {
            let br = ty::BoundRegion { var: ty::BoundVar::from_u32(counter), kind: ty::BrAnon };
            counter += 1;
            ty::Region::new_bound(tcx, current_depth, br)
        }
        // All free regions should be erased here.
        r => bug!("unexpected region: {r:?}"),
    });
    let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
        (0..counter).map(|_| ty::BoundVariableKind::Region(ty::BrAnon)),
    );
    ty::Binder::bind_with_vars(ty, bound_vars)
}

#[instrument(level = "debug", skip(ecx), ret)]
pub(in crate::solve) fn instantiate_constituent_tys_for_sized_trait<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<ty::Binder<'tcx, Ty<'tcx>>>, NoSolution> {
    match *ty.kind() {
        // impl Sized for u*, i*, bool, f*, FnDef, FnPtr, *(const/mut) T, char, &mut? T, [T; N], dyn* Trait, !
        // impl Sized for Coroutine, CoroutineWitness, Closure, CoroutineClosure
        ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::Uint(_)
        | ty::Int(_)
        | ty::Bool
        | ty::Float(_)
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::RawPtr(..)
        | ty::Char
        | ty::Ref(..)
        | ty::Coroutine(..)
        | ty::CoroutineWitness(..)
        | ty::Array(..)
        | ty::Closure(..)
        | ty::CoroutineClosure(..)
        | ty::Never
        | ty::Dynamic(_, _, ty::DynStar)
        | ty::Error(_) => Ok(vec![]),

        ty::Str
        | ty::Slice(_)
        | ty::Dynamic(..)
        | ty::Foreign(..)
        | ty::Alias(..)
        | ty::Param(_)
        | ty::Placeholder(..) => Err(NoSolution),

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
            bug!("unexpected type `{ty}`")
        }

        // impl Sized for (T1, T2, .., Tn) where T1: Sized, T2: Sized, .. Tn: Sized
        ty::Tuple(tys) => Ok(tys.iter().map(ty::Binder::dummy).collect()),

        // impl Sized for Adt where T: Sized forall T in field types
        ty::Adt(def, args) => {
            let sized_crit = def.sized_constraint(ecx.tcx());
            Ok(sized_crit.iter_instantiated(ecx.tcx(), args).map(ty::Binder::dummy).collect())
        }
    }
}

#[instrument(level = "debug", skip(ecx), ret)]
pub(in crate::solve) fn instantiate_constituent_tys_for_copy_clone_trait<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    ty: Ty<'tcx>,
) -> Result<Vec<ty::Binder<'tcx, Ty<'tcx>>>, NoSolution> {
    match *ty.kind() {
        // impl Copy/Clone for FnDef, FnPtr
        ty::FnDef(..) | ty::FnPtr(_) | ty::Error(_) => Ok(vec![]),

        // Implementations are provided in core
        ty::Uint(_)
        | ty::Int(_)
        | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::Bool
        | ty::Float(_)
        | ty::Char
        | ty::RawPtr(..)
        | ty::Never
        | ty::Ref(_, _, Mutability::Not)
        | ty::Array(..) => Err(NoSolution),

        ty::Dynamic(..)
        | ty::Str
        | ty::Slice(_)
        | ty::Foreign(..)
        | ty::Ref(_, _, Mutability::Mut)
        | ty::Adt(_, _)
        | ty::Alias(_, _)
        | ty::Param(_)
        | ty::Placeholder(..) => Err(NoSolution),

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
            bug!("unexpected type `{ty}`")
        }

        // impl Copy/Clone for (T1, T2, .., Tn) where T1: Copy/Clone, T2: Copy/Clone, .. Tn: Copy/Clone
        ty::Tuple(tys) => Ok(tys.iter().map(ty::Binder::dummy).collect()),

        // impl Copy/Clone for Closure where Self::TupledUpvars: Copy/Clone
        ty::Closure(_, args) => Ok(vec![ty::Binder::dummy(args.as_closure().tupled_upvars_ty())]),

        ty::CoroutineClosure(..) => Err(NoSolution),

        // only when `coroutine_clone` is enabled and the coroutine is movable
        // impl Copy/Clone for Coroutine where T: Copy/Clone forall T in (upvars, witnesses)
        ty::Coroutine(def_id, args) => match ecx.tcx().coroutine_movability(def_id) {
            Movability::Static => Err(NoSolution),
            Movability::Movable => {
                if ecx.tcx().features().coroutine_clone {
                    let coroutine = args.as_coroutine();
                    Ok(vec![
                        ty::Binder::dummy(coroutine.tupled_upvars_ty()),
                        ty::Binder::dummy(coroutine.witness()),
                    ])
                } else {
                    Err(NoSolution)
                }
            }
        },

        // impl Copy/Clone for CoroutineWitness where T: Copy/Clone forall T in coroutine_hidden_types
        ty::CoroutineWitness(def_id, args) => Ok(ecx
            .tcx()
            .coroutine_hidden_types(def_id)
            .map(|bty| {
                replace_erased_lifetimes_with_bound_vars(
                    ecx.tcx(),
                    bty.instantiate(ecx.tcx(), args),
                )
            })
            .collect()),
    }
}

// Returns a binder of the tupled inputs types and output type from a builtin callable type.
pub(in crate::solve) fn extract_tupled_inputs_and_output_from_callable<'tcx>(
    tcx: TyCtxt<'tcx>,
    self_ty: Ty<'tcx>,
    goal_kind: ty::ClosureKind,
) -> Result<Option<ty::Binder<'tcx, (Ty<'tcx>, Ty<'tcx>)>>, NoSolution> {
    match *self_ty.kind() {
        // keep this in sync with assemble_fn_pointer_candidates until the old solver is removed.
        ty::FnDef(def_id, args) => {
            let sig = tcx.fn_sig(def_id);
            if sig.skip_binder().is_fn_trait_compatible()
                && tcx.codegen_fn_attrs(def_id).target_features.is_empty()
            {
                Ok(Some(
                    sig.instantiate(tcx, args)
                        .map_bound(|sig| (Ty::new_tup(tcx, sig.inputs()), sig.output())),
                ))
            } else {
                Err(NoSolution)
            }
        }
        // keep this in sync with assemble_fn_pointer_candidates until the old solver is removed.
        ty::FnPtr(sig) => {
            if sig.is_fn_trait_compatible() {
                Ok(Some(sig.map_bound(|sig| (Ty::new_tup(tcx, sig.inputs()), sig.output()))))
            } else {
                Err(NoSolution)
            }
        }
        ty::Closure(_, args) => {
            let closure_args = args.as_closure();
            match closure_args.kind_ty().to_opt_closure_kind() {
                // If the closure's kind doesn't extend the goal kind,
                // then the closure doesn't implement the trait.
                Some(closure_kind) => {
                    if !closure_kind.extends(goal_kind) {
                        return Err(NoSolution);
                    }
                }
                // Closure kind is not yet determined, so we return ambiguity unless
                // the expected kind is `FnOnce` as that is always implemented.
                None => {
                    if goal_kind != ty::ClosureKind::FnOnce {
                        return Ok(None);
                    }
                }
            }
            Ok(Some(closure_args.sig().map_bound(|sig| (sig.inputs()[0], sig.output()))))
        }

        // Coroutine-closures don't implement `Fn` traits the normal way.
        // Instead, they always implement `FnOnce`, but only implement
        // `FnMut`/`Fn` if they capture no upvars, since those may borrow
        // from the closure.
        ty::CoroutineClosure(def_id, args) => {
            let args = args.as_coroutine_closure();
            let kind_ty = args.kind_ty();
            let sig = args.coroutine_closure_sig().skip_binder();

            let coroutine_ty = if let Some(closure_kind) = kind_ty.to_opt_closure_kind() {
                if !closure_kind.extends(goal_kind) {
                    return Err(NoSolution);
                }

                // If `Fn`/`FnMut`, we only implement this goal if we
                // have no captures.
                let no_borrows = match args.tupled_upvars_ty().kind() {
                    ty::Tuple(tys) => tys.is_empty(),
                    ty::Error(_) => false,
                    _ => bug!("tuple_fields called on non-tuple"),
                };
                if closure_kind != ty::ClosureKind::FnOnce && !no_borrows {
                    return Err(NoSolution);
                }

                coroutine_closure_to_certain_coroutine(
                    tcx,
                    goal_kind,
                    // No captures by ref, so this doesn't matter.
                    tcx.lifetimes.re_static,
                    def_id,
                    args,
                    sig,
                )
            } else {
                // Closure kind is not yet determined, so we return ambiguity unless
                // the expected kind is `FnOnce` as that is always implemented.
                if goal_kind != ty::ClosureKind::FnOnce {
                    return Ok(None);
                }

                coroutine_closure_to_ambiguous_coroutine(
                    tcx,
                    goal_kind, // No captures by ref, so this doesn't matter.
                    tcx.lifetimes.re_static,
                    def_id,
                    args,
                    sig,
                )
            };

            Ok(Some(args.coroutine_closure_sig().rebind((sig.tupled_inputs_ty, coroutine_ty))))
        }

        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Adt(_, _)
        | ty::Foreign(_)
        | ty::Str
        | ty::Array(_, _)
        | ty::Slice(_)
        | ty::RawPtr(_)
        | ty::Ref(_, _, _)
        | ty::Dynamic(_, _, _)
        | ty::Coroutine(_, _)
        | ty::CoroutineWitness(..)
        | ty::Never
        | ty::Tuple(_)
        | ty::Alias(_, _)
        | ty::Param(_)
        | ty::Placeholder(..)
        | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::Error(_) => Err(NoSolution),

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
            bug!("unexpected type `{self_ty}`")
        }
    }
}

/// Relevant types for an async callable, including its inputs, output,
/// and the return type you get from awaiting the output.
#[derive(Copy, Clone, Debug, TypeVisitable, TypeFoldable)]
pub(in crate::solve) struct AsyncCallableRelevantTypes<'tcx> {
    pub tupled_inputs_ty: Ty<'tcx>,
    /// Type returned by calling the closure
    /// i.e. `f()`.
    pub output_coroutine_ty: Ty<'tcx>,
    /// Type returned by `await`ing the output
    /// i.e. `f().await`.
    pub coroutine_return_ty: Ty<'tcx>,
}

// Returns a binder of the tupled inputs types, output type, and coroutine type
// from a builtin coroutine-closure type. If we don't yet know the closure kind of
// the coroutine-closure, emit an additional trait predicate for `AsyncFnKindHelper`
// which enforces the closure is actually callable with the given trait. When we
// know the kind already, we can short-circuit this check.
pub(in crate::solve) fn extract_tupled_inputs_and_output_from_async_callable<'tcx>(
    tcx: TyCtxt<'tcx>,
    self_ty: Ty<'tcx>,
    goal_kind: ty::ClosureKind,
    env_region: ty::Region<'tcx>,
) -> Result<
    (ty::Binder<'tcx, AsyncCallableRelevantTypes<'tcx>>, Vec<ty::Predicate<'tcx>>),
    NoSolution,
> {
    match *self_ty.kind() {
        ty::CoroutineClosure(def_id, args) => {
            let args = args.as_coroutine_closure();
            let kind_ty = args.kind_ty();
            let sig = args.coroutine_closure_sig().skip_binder();
            let mut nested = vec![];
            let coroutine_ty = if let Some(closure_kind) = kind_ty.to_opt_closure_kind() {
                if !closure_kind.extends(goal_kind) {
                    return Err(NoSolution);
                }

                coroutine_closure_to_certain_coroutine(
                    tcx, goal_kind, env_region, def_id, args, sig,
                )
            } else {
                // When we don't know the closure kind (and therefore also the closure's upvars,
                // which are computed at the same time), we must delay the computation of the
                // generator's upvars. We do this using the `AsyncFnKindHelper`, which as a trait
                // goal functions similarly to the old `ClosureKind` predicate, and ensures that
                // the goal kind <= the closure kind. As a projection `AsyncFnKindHelper::Upvars`
                // will project to the right upvars for the generator, appending the inputs and
                // coroutine upvars respecting the closure kind.
                nested.push(
                    ty::TraitRef::new(
                        tcx,
                        tcx.require_lang_item(LangItem::AsyncFnKindHelper, None),
                        [kind_ty, Ty::from_closure_kind(tcx, goal_kind)],
                    )
                    .to_predicate(tcx),
                );

                coroutine_closure_to_ambiguous_coroutine(
                    tcx, goal_kind, env_region, def_id, args, sig,
                )
            };

            Ok((
                args.coroutine_closure_sig().rebind(AsyncCallableRelevantTypes {
                    tupled_inputs_ty: sig.tupled_inputs_ty,
                    output_coroutine_ty: coroutine_ty,
                    coroutine_return_ty: sig.return_ty,
                }),
                nested,
            ))
        }

        ty::FnDef(..) | ty::FnPtr(..) => {
            let bound_sig = self_ty.fn_sig(tcx);
            let sig = bound_sig.skip_binder();
            let future_trait_def_id = tcx.require_lang_item(LangItem::Future, None);
            // `FnDef` and `FnPtr` only implement `AsyncFn*` when their
            // return type implements `Future`.
            let nested = vec![
                bound_sig
                    .rebind(ty::TraitRef::new(tcx, future_trait_def_id, [sig.output()]))
                    .to_predicate(tcx),
            ];
            let future_output_def_id = tcx
                .associated_items(future_trait_def_id)
                .filter_by_name_unhygienic(sym::Output)
                .next()
                .unwrap()
                .def_id;
            let future_output_ty = Ty::new_projection(tcx, future_output_def_id, [sig.output()]);
            Ok((
                bound_sig.rebind(AsyncCallableRelevantTypes {
                    tupled_inputs_ty: Ty::new_tup(tcx, sig.inputs()),
                    output_coroutine_ty: sig.output(),
                    coroutine_return_ty: future_output_ty,
                }),
                nested,
            ))
        }
        ty::Closure(_, args) => {
            let args = args.as_closure();
            let bound_sig = args.sig();
            let sig = bound_sig.skip_binder();
            let future_trait_def_id = tcx.require_lang_item(LangItem::Future, None);
            // `Closure`s only implement `AsyncFn*` when their return type
            // implements `Future`.
            let mut nested = vec![
                bound_sig
                    .rebind(ty::TraitRef::new(tcx, future_trait_def_id, [sig.output()]))
                    .to_predicate(tcx),
            ];

            // Additionally, we need to check that the closure kind
            // is still compatible.
            let kind_ty = args.kind_ty();
            if let Some(closure_kind) = kind_ty.to_opt_closure_kind() {
                if !closure_kind.extends(goal_kind) {
                    return Err(NoSolution);
                }
            } else {
                let async_fn_kind_trait_def_id =
                    tcx.require_lang_item(LangItem::AsyncFnKindHelper, None);
                // When we don't know the closure kind (and therefore also the closure's upvars,
                // which are computed at the same time), we must delay the computation of the
                // generator's upvars. We do this using the `AsyncFnKindHelper`, which as a trait
                // goal functions similarly to the old `ClosureKind` predicate, and ensures that
                // the goal kind <= the closure kind. As a projection `AsyncFnKindHelper::Upvars`
                // will project to the right upvars for the generator, appending the inputs and
                // coroutine upvars respecting the closure kind.
                nested.push(
                    ty::TraitRef::new(
                        tcx,
                        async_fn_kind_trait_def_id,
                        [kind_ty, Ty::from_closure_kind(tcx, goal_kind)],
                    )
                    .to_predicate(tcx),
                );
            }

            let future_output_def_id = tcx
                .associated_items(future_trait_def_id)
                .filter_by_name_unhygienic(sym::Output)
                .next()
                .unwrap()
                .def_id;
            let future_output_ty = Ty::new_projection(tcx, future_output_def_id, [sig.output()]);
            Ok((
                bound_sig.rebind(AsyncCallableRelevantTypes {
                    tupled_inputs_ty: sig.inputs()[0],
                    output_coroutine_ty: sig.output(),
                    coroutine_return_ty: future_output_ty,
                }),
                nested,
            ))
        }

        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Adt(_, _)
        | ty::Foreign(_)
        | ty::Str
        | ty::Array(_, _)
        | ty::Slice(_)
        | ty::RawPtr(_)
        | ty::Ref(_, _, _)
        | ty::Dynamic(_, _, _)
        | ty::Coroutine(_, _)
        | ty::CoroutineWitness(..)
        | ty::Never
        | ty::Tuple(_)
        | ty::Alias(_, _)
        | ty::Param(_)
        | ty::Placeholder(..)
        | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::Error(_) => Err(NoSolution),

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
            bug!("unexpected type `{self_ty}`")
        }
    }
}

/// Given a coroutine-closure, project to its returned coroutine when we are *certain*
/// that the closure's kind is compatible with the goal.
fn coroutine_closure_to_certain_coroutine<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal_kind: ty::ClosureKind,
    goal_region: ty::Region<'tcx>,
    def_id: DefId,
    args: ty::CoroutineClosureArgs<'tcx>,
    sig: ty::CoroutineClosureSignature<'tcx>,
) -> Ty<'tcx> {
    sig.to_coroutine_given_kind_and_upvars(
        tcx,
        args.parent_args(),
        tcx.coroutine_for_closure(def_id),
        goal_kind,
        goal_region,
        args.tupled_upvars_ty(),
        args.coroutine_captures_by_ref_ty(),
    )
}

/// Given a coroutine-closure, project to its returned coroutine when we are *not certain*
/// that the closure's kind is compatible with the goal, and therefore also don't know
/// yet what the closure's upvars are.
///
/// Note that we do not also push a `AsyncFnKindHelper` goal here.
fn coroutine_closure_to_ambiguous_coroutine<'tcx>(
    tcx: TyCtxt<'tcx>,
    goal_kind: ty::ClosureKind,
    goal_region: ty::Region<'tcx>,
    def_id: DefId,
    args: ty::CoroutineClosureArgs<'tcx>,
    sig: ty::CoroutineClosureSignature<'tcx>,
) -> Ty<'tcx> {
    let async_fn_kind_trait_def_id = tcx.require_lang_item(LangItem::AsyncFnKindHelper, None);
    let upvars_projection_def_id = tcx
        .associated_items(async_fn_kind_trait_def_id)
        .filter_by_name_unhygienic(sym::Upvars)
        .next()
        .unwrap()
        .def_id;
    let tupled_upvars_ty = Ty::new_projection(
        tcx,
        upvars_projection_def_id,
        [
            ty::GenericArg::from(args.kind_ty()),
            Ty::from_closure_kind(tcx, goal_kind).into(),
            goal_region.into(),
            sig.tupled_inputs_ty.into(),
            args.tupled_upvars_ty().into(),
            args.coroutine_captures_by_ref_ty().into(),
        ],
    );
    sig.to_coroutine(
        tcx,
        args.parent_args(),
        Ty::from_closure_kind(tcx, goal_kind),
        tcx.coroutine_for_closure(def_id),
        tupled_upvars_ty,
    )
}

/// Assemble a list of predicates that would be present on a theoretical
/// user impl for an object type. These predicates must be checked any time
/// we assemble a built-in object candidate for an object type, since they
/// are not implied by the well-formedness of the type.
///
/// For example, given the following traits:
///
/// ```rust,ignore (theoretical code)
/// trait Foo: Baz {
///     type Bar: Copy;
/// }
///
/// trait Baz {}
/// ```
///
/// For the dyn type `dyn Foo<Item = Ty>`, we can imagine there being a
/// pair of theoretical impls:
///
/// ```rust,ignore (theoretical code)
/// impl Foo for dyn Foo<Item = Ty>
/// where
///     Self: Baz,
///     <Self as Foo>::Bar: Copy,
/// {
///     type Bar = Ty;
/// }
///
/// impl Baz for dyn Foo<Item = Ty> {}
/// ```
///
/// However, in order to make such impls well-formed, we need to do an
/// additional step of eagerly folding the associated types in the where
/// clauses of the impl. In this example, that means replacing
/// `<Self as Foo>::Bar` with `Ty` in the first impl.
///
// FIXME: This is only necessary as `<Self as Trait>::Assoc: ItemBound`
// bounds in impls are trivially proven using the item bound candidates.
// This is unsound in general and once that is fixed, we don't need to
// normalize eagerly here. See https://github.com/lcnr/solver-woes/issues/9
// for more details.
pub(in crate::solve) fn predicates_for_object_candidate<'tcx>(
    ecx: &EvalCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    object_bound: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> Vec<Goal<'tcx, ty::Predicate<'tcx>>> {
    let tcx = ecx.tcx();
    let mut requirements = vec![];
    requirements.extend(
        tcx.super_predicates_of(trait_ref.def_id).instantiate(tcx, trait_ref.args).predicates,
    );
    for item in tcx.associated_items(trait_ref.def_id).in_definition_order() {
        // FIXME(associated_const_equality): Also add associated consts to
        // the requirements here.
        if item.kind == ty::AssocKind::Type {
            // associated types that require `Self: Sized` do not show up in the built-in
            // implementation of `Trait for dyn Trait`, and can be dropped here.
            if tcx.generics_require_sized_self(item.def_id) {
                continue;
            }

            requirements
                .extend(tcx.item_bounds(item.def_id).iter_instantiated(tcx, trait_ref.args));
        }
    }

    let mut replace_projection_with = FxHashMap::default();
    for bound in object_bound {
        if let ty::ExistentialPredicate::Projection(proj) = bound.skip_binder() {
            let proj = proj.with_self_ty(tcx, trait_ref.self_ty());
            let old_ty = replace_projection_with.insert(proj.def_id(), bound.rebind(proj));
            assert_eq!(
                old_ty,
                None,
                "{} has two generic parameters: {} and {}",
                proj.projection_ty,
                proj.term,
                old_ty.unwrap()
            );
        }
    }

    let mut folder =
        ReplaceProjectionWith { ecx, param_env, mapping: replace_projection_with, nested: vec![] };
    let folded_requirements = requirements.fold_with(&mut folder);

    folder
        .nested
        .into_iter()
        .chain(folded_requirements.into_iter().map(|clause| Goal::new(tcx, param_env, clause)))
        .collect()
}

struct ReplaceProjectionWith<'a, 'tcx> {
    ecx: &'a EvalCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    mapping: FxHashMap<DefId, ty::PolyProjectionPredicate<'tcx>>,
    nested: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ReplaceProjectionWith<'_, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.ecx.tcx()
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Alias(ty::Projection, alias_ty) = *ty.kind()
            && let Some(replacement) = self.mapping.get(&alias_ty.def_id)
        {
            // We may have a case where our object type's projection bound is higher-ranked,
            // but the where clauses we instantiated are not. We can solve this by instantiating
            // the binder at the usage site.
            let proj = self.ecx.instantiate_binder_with_infer(*replacement);
            // FIXME: Technically this equate could be fallible...
            self.nested.extend(
                self.ecx
                    .eq_and_get_goals(self.param_env, alias_ty, proj.projection_ty)
                    .expect("expected to be able to unify goal projection with dyn's projection"),
            );
            proj.term.ty().unwrap()
        } else {
            ty.super_fold_with(self)
        }
    }
}
