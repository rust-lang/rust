//! Code which is used by built-in goals that match "structurally", such a auto
//! traits, `Copy`/`Clone`.

use derive_where::derive_where;
use rustc_ast_ir::{Movability, Mutability};
use rustc_type_ir::data_structures::HashMap;
use rustc_type_ir::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_type_ir::inherent::*;
use rustc_type_ir::lang_items::TraitSolverLangItem;
use rustc_type_ir::{self as ty, Interner, Upcast as _};
use rustc_type_ir_macros::{TypeFoldable_Generic, TypeVisitable_Generic};
use tracing::instrument;

use crate::delegate::SolverDelegate;
use crate::solve::{EvalCtxt, Goal, NoSolution};

// Calculates the constituent types of a type for `auto trait` purposes.
#[instrument(level = "trace", skip(ecx), ret)]
pub(in crate::solve) fn instantiate_constituent_tys_for_auto_trait<D, I>(
    ecx: &EvalCtxt<'_, D>,
    ty: I::Ty,
) -> Result<Vec<ty::Binder<I, I::Ty>>, NoSolution>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    let cx = ecx.cx();
    match ty.kind() {
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
        ty::Str => Ok(vec![ty::Binder::dummy(Ty::new_slice(cx, Ty::new_u8(cx)))]),

        ty::Dynamic(..)
        | ty::Param(..)
        | ty::Foreign(..)
        | ty::Alias(ty::Projection | ty::Inherent | ty::Weak, ..)
        | ty::Placeholder(..)
        | ty::Bound(..)
        | ty::Infer(_) => {
            panic!("unexpected type `{ty:?}`")
        }

        ty::RawPtr(element_ty, _) | ty::Ref(_, element_ty, _) => {
            Ok(vec![ty::Binder::dummy(element_ty)])
        }

        ty::Pat(element_ty, _) | ty::Array(element_ty, _) | ty::Slice(element_ty) => {
            Ok(vec![ty::Binder::dummy(element_ty)])
        }

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
            .cx()
            .bound_coroutine_hidden_types(def_id)
            .into_iter()
            .map(|bty| bty.instantiate(cx, args))
            .collect()),

        // For `PhantomData<T>`, we pass `T`.
        ty::Adt(def, args) if def.is_phantom_data() => Ok(vec![ty::Binder::dummy(args.type_at(0))]),

        ty::Adt(def, args) => {
            Ok(def.all_field_tys(cx).iter_instantiated(cx, args).map(ty::Binder::dummy).collect())
        }

        ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) => {
            // We can resolve the `impl Trait` to its concrete type,
            // which enforces a DAG between the functions requiring
            // the auto trait bounds in question.
            Ok(vec![ty::Binder::dummy(cx.type_of(def_id).instantiate(cx, args))])
        }
    }
}

#[instrument(level = "trace", skip(ecx), ret)]
pub(in crate::solve) fn instantiate_constituent_tys_for_sized_trait<D, I>(
    ecx: &EvalCtxt<'_, D>,
    ty: I::Ty,
) -> Result<Vec<ty::Binder<I, I::Ty>>, NoSolution>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    match ty.kind() {
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
        | ty::Pat(..)
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
            panic!("unexpected type `{ty:?}`")
        }

        // impl Sized for ()
        // impl Sized for (T1, T2, .., Tn) where Tn: Sized if n >= 1
        ty::Tuple(tys) => Ok(tys.last().map_or_else(Vec::new, |ty| vec![ty::Binder::dummy(ty)])),

        // impl Sized for Adt<Args...> where sized_constraint(Adt)<Args...>: Sized
        //   `sized_constraint(Adt)` is the deepest struct trail that can be determined
        //   by the definition of `Adt`, independent of the generic args.
        // impl Sized for Adt<Args...> if sized_constraint(Adt) == None
        //   As a performance optimization, `sized_constraint(Adt)` can return `None`
        //   if the ADTs definition implies that it is sized by for all possible args.
        //   In this case, the builtin impl will have no nested subgoals. This is a
        //   "best effort" optimization and `sized_constraint` may return `Some`, even
        //   if the ADT is sized for all possible args.
        ty::Adt(def, args) => {
            if let Some(sized_crit) = def.sized_constraint(ecx.cx()) {
                Ok(vec![ty::Binder::dummy(sized_crit.instantiate(ecx.cx(), args))])
            } else {
                Ok(vec![])
            }
        }
    }
}

#[instrument(level = "trace", skip(ecx), ret)]
pub(in crate::solve) fn instantiate_constituent_tys_for_copy_clone_trait<D, I>(
    ecx: &EvalCtxt<'_, D>,
    ty: I::Ty,
) -> Result<Vec<ty::Binder<I, I::Ty>>, NoSolution>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    match ty.kind() {
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

        // Cannot implement in core, as we can't be generic over patterns yet,
        // so we'd have to list all patterns and type combinations.
        ty::Pat(ty, ..) => Ok(vec![ty::Binder::dummy(ty)]),

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
            panic!("unexpected type `{ty:?}`")
        }

        // impl Copy/Clone for (T1, T2, .., Tn) where T1: Copy/Clone, T2: Copy/Clone, .. Tn: Copy/Clone
        ty::Tuple(tys) => Ok(tys.iter().map(ty::Binder::dummy).collect()),

        // impl Copy/Clone for Closure where Self::TupledUpvars: Copy/Clone
        ty::Closure(_, args) => Ok(vec![ty::Binder::dummy(args.as_closure().tupled_upvars_ty())]),

        ty::CoroutineClosure(..) => Err(NoSolution),

        // only when `coroutine_clone` is enabled and the coroutine is movable
        // impl Copy/Clone for Coroutine where T: Copy/Clone forall T in (upvars, witnesses)
        ty::Coroutine(def_id, args) => match ecx.cx().coroutine_movability(def_id) {
            Movability::Static => Err(NoSolution),
            Movability::Movable => {
                if ecx.cx().features().coroutine_clone() {
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
            .cx()
            .bound_coroutine_hidden_types(def_id)
            .into_iter()
            .map(|bty| bty.instantiate(ecx.cx(), args))
            .collect()),
    }
}

// Returns a binder of the tupled inputs types and output type from a builtin callable type.
pub(in crate::solve) fn extract_tupled_inputs_and_output_from_callable<I: Interner>(
    cx: I,
    self_ty: I::Ty,
    goal_kind: ty::ClosureKind,
) -> Result<Option<ty::Binder<I, (I::Ty, I::Ty)>>, NoSolution> {
    match self_ty.kind() {
        // keep this in sync with assemble_fn_pointer_candidates until the old solver is removed.
        ty::FnDef(def_id, args) => {
            let sig = cx.fn_sig(def_id);
            if sig.skip_binder().is_fn_trait_compatible() && !cx.has_target_features(def_id) {
                Ok(Some(
                    sig.instantiate(cx, args)
                        .map_bound(|sig| (Ty::new_tup(cx, sig.inputs().as_slice()), sig.output())),
                ))
            } else {
                Err(NoSolution)
            }
        }
        // keep this in sync with assemble_fn_pointer_candidates until the old solver is removed.
        ty::FnPtr(sig) => {
            if sig.is_fn_trait_compatible() {
                Ok(Some(
                    sig.map_bound(|sig| (Ty::new_tup(cx, sig.inputs().as_slice()), sig.output())),
                ))
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
            Ok(Some(
                closure_args.sig().map_bound(|sig| (sig.inputs().get(0).unwrap(), sig.output())),
            ))
        }

        // Coroutine-closures don't implement `Fn` traits the normal way.
        // Instead, they always implement `FnOnce`, but only implement
        // `FnMut`/`Fn` if they capture no upvars, since those may borrow
        // from the closure.
        ty::CoroutineClosure(def_id, args) => {
            let args = args.as_coroutine_closure();
            let kind_ty = args.kind_ty();
            let sig = args.coroutine_closure_sig().skip_binder();

            // FIXME: let_chains
            let kind = kind_ty.to_opt_closure_kind();
            let coroutine_ty = if kind.is_some() && !args.tupled_upvars_ty().is_ty_var() {
                let closure_kind = kind.unwrap();
                if !closure_kind.extends(goal_kind) {
                    return Err(NoSolution);
                }

                // A coroutine-closure implements `FnOnce` *always*, since it may
                // always be called once. It additionally implements `Fn`/`FnMut`
                // only if it has no upvars referencing the closure-env lifetime,
                // and if the closure kind permits it.
                if closure_kind != ty::ClosureKind::FnOnce && args.has_self_borrows() {
                    return Err(NoSolution);
                }

                coroutine_closure_to_certain_coroutine(
                    cx,
                    goal_kind,
                    // No captures by ref, so this doesn't matter.
                    Region::new_static(cx),
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
                    cx,
                    goal_kind, // No captures by ref, so this doesn't matter.
                    Region::new_static(cx),
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
        | ty::RawPtr(_, _)
        | ty::Ref(_, _, _)
        | ty::Dynamic(_, _, _)
        | ty::Coroutine(_, _)
        | ty::CoroutineWitness(..)
        | ty::Never
        | ty::Tuple(_)
        | ty::Pat(_, _)
        | ty::Alias(_, _)
        | ty::Param(_)
        | ty::Placeholder(..)
        | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
        | ty::Error(_) => Err(NoSolution),

        ty::Bound(..)
        | ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
            panic!("unexpected type `{self_ty:?}`")
        }
    }
}

/// Relevant types for an async callable, including its inputs, output,
/// and the return type you get from awaiting the output.
#[derive_where(Clone, Copy, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
pub(in crate::solve) struct AsyncCallableRelevantTypes<I: Interner> {
    pub tupled_inputs_ty: I::Ty,
    /// Type returned by calling the closure
    /// i.e. `f()`.
    pub output_coroutine_ty: I::Ty,
    /// Type returned by `await`ing the output
    /// i.e. `f().await`.
    pub coroutine_return_ty: I::Ty,
}

// Returns a binder of the tupled inputs types, output type, and coroutine type
// from a builtin coroutine-closure type. If we don't yet know the closure kind of
// the coroutine-closure, emit an additional trait predicate for `AsyncFnKindHelper`
// which enforces the closure is actually callable with the given trait. When we
// know the kind already, we can short-circuit this check.
pub(in crate::solve) fn extract_tupled_inputs_and_output_from_async_callable<I: Interner>(
    cx: I,
    self_ty: I::Ty,
    goal_kind: ty::ClosureKind,
    env_region: I::Region,
) -> Result<(ty::Binder<I, AsyncCallableRelevantTypes<I>>, Vec<I::Predicate>), NoSolution> {
    match self_ty.kind() {
        ty::CoroutineClosure(def_id, args) => {
            let args = args.as_coroutine_closure();
            let kind_ty = args.kind_ty();
            let sig = args.coroutine_closure_sig().skip_binder();
            let mut nested = vec![];

            // FIXME: let_chains
            let kind = kind_ty.to_opt_closure_kind();
            let coroutine_ty = if kind.is_some() && !args.tupled_upvars_ty().is_ty_var() {
                if !kind.unwrap().extends(goal_kind) {
                    return Err(NoSolution);
                }

                coroutine_closure_to_certain_coroutine(cx, goal_kind, env_region, def_id, args, sig)
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
                        cx,
                        cx.require_lang_item(TraitSolverLangItem::AsyncFnKindHelper),
                        [kind_ty, Ty::from_closure_kind(cx, goal_kind)],
                    )
                    .upcast(cx),
                );

                coroutine_closure_to_ambiguous_coroutine(
                    cx, goal_kind, env_region, def_id, args, sig,
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
            let bound_sig = self_ty.fn_sig(cx);
            let sig = bound_sig.skip_binder();
            let future_trait_def_id = cx.require_lang_item(TraitSolverLangItem::Future);
            // `FnDef` and `FnPtr` only implement `AsyncFn*` when their
            // return type implements `Future`.
            let nested = vec![
                bound_sig
                    .rebind(ty::TraitRef::new(cx, future_trait_def_id, [sig.output()]))
                    .upcast(cx),
            ];
            let future_output_def_id = cx.require_lang_item(TraitSolverLangItem::FutureOutput);
            let future_output_ty = Ty::new_projection(cx, future_output_def_id, [sig.output()]);
            Ok((
                bound_sig.rebind(AsyncCallableRelevantTypes {
                    tupled_inputs_ty: Ty::new_tup(cx, sig.inputs().as_slice()),
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
            let future_trait_def_id = cx.require_lang_item(TraitSolverLangItem::Future);
            // `Closure`s only implement `AsyncFn*` when their return type
            // implements `Future`.
            let mut nested = vec![
                bound_sig
                    .rebind(ty::TraitRef::new(cx, future_trait_def_id, [sig.output()]))
                    .upcast(cx),
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
                    cx.require_lang_item(TraitSolverLangItem::AsyncFnKindHelper);
                // When we don't know the closure kind (and therefore also the closure's upvars,
                // which are computed at the same time), we must delay the computation of the
                // generator's upvars. We do this using the `AsyncFnKindHelper`, which as a trait
                // goal functions similarly to the old `ClosureKind` predicate, and ensures that
                // the goal kind <= the closure kind. As a projection `AsyncFnKindHelper::Upvars`
                // will project to the right upvars for the generator, appending the inputs and
                // coroutine upvars respecting the closure kind.
                nested.push(
                    ty::TraitRef::new(
                        cx,
                        async_fn_kind_trait_def_id,
                        [kind_ty, Ty::from_closure_kind(cx, goal_kind)],
                    )
                    .upcast(cx),
                );
            }

            let future_output_def_id = cx.require_lang_item(TraitSolverLangItem::FutureOutput);
            let future_output_ty = Ty::new_projection(cx, future_output_def_id, [sig.output()]);
            Ok((
                bound_sig.rebind(AsyncCallableRelevantTypes {
                    tupled_inputs_ty: sig.inputs().get(0).unwrap(),
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
        | ty::Pat(_, _)
        | ty::Slice(_)
        | ty::RawPtr(_, _)
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
            panic!("unexpected type `{self_ty:?}`")
        }
    }
}

/// Given a coroutine-closure, project to its returned coroutine when we are *certain*
/// that the closure's kind is compatible with the goal.
fn coroutine_closure_to_certain_coroutine<I: Interner>(
    cx: I,
    goal_kind: ty::ClosureKind,
    goal_region: I::Region,
    def_id: I::DefId,
    args: ty::CoroutineClosureArgs<I>,
    sig: ty::CoroutineClosureSignature<I>,
) -> I::Ty {
    sig.to_coroutine_given_kind_and_upvars(
        cx,
        args.parent_args(),
        cx.coroutine_for_closure(def_id),
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
fn coroutine_closure_to_ambiguous_coroutine<I: Interner>(
    cx: I,
    goal_kind: ty::ClosureKind,
    goal_region: I::Region,
    def_id: I::DefId,
    args: ty::CoroutineClosureArgs<I>,
    sig: ty::CoroutineClosureSignature<I>,
) -> I::Ty {
    let upvars_projection_def_id = cx.require_lang_item(TraitSolverLangItem::AsyncFnKindUpvars);
    let tupled_upvars_ty = Ty::new_projection(
        cx,
        upvars_projection_def_id,
        [
            I::GenericArg::from(args.kind_ty()),
            Ty::from_closure_kind(cx, goal_kind).into(),
            goal_region.into(),
            sig.tupled_inputs_ty.into(),
            args.tupled_upvars_ty().into(),
            args.coroutine_captures_by_ref_ty().into(),
        ],
    );
    sig.to_coroutine(
        cx,
        args.parent_args(),
        Ty::from_closure_kind(cx, goal_kind),
        cx.coroutine_for_closure(def_id),
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
pub(in crate::solve) fn predicates_for_object_candidate<D, I>(
    ecx: &EvalCtxt<'_, D>,
    param_env: I::ParamEnv,
    trait_ref: ty::TraitRef<I>,
    object_bounds: I::BoundExistentialPredicates,
) -> Vec<Goal<I, I::Predicate>>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    let cx = ecx.cx();
    let mut requirements = vec![];
    requirements.extend(
        cx.explicit_super_predicates_of(trait_ref.def_id)
            .iter_instantiated(cx, trait_ref.args)
            .map(|(pred, _)| pred),
    );

    // FIXME(associated_const_equality): Also add associated consts to
    // the requirements here.
    for associated_type_def_id in cx.associated_type_def_ids(trait_ref.def_id) {
        // associated types that require `Self: Sized` do not show up in the built-in
        // implementation of `Trait for dyn Trait`, and can be dropped here.
        if cx.generics_require_sized_self(associated_type_def_id) {
            continue;
        }

        requirements
            .extend(cx.item_bounds(associated_type_def_id).iter_instantiated(cx, trait_ref.args));
    }

    let mut replace_projection_with = HashMap::default();
    for bound in object_bounds.iter() {
        if let ty::ExistentialPredicate::Projection(proj) = bound.skip_binder() {
            let proj = proj.with_self_ty(cx, trait_ref.self_ty());
            let old_ty = replace_projection_with.insert(proj.def_id(), bound.rebind(proj));
            assert_eq!(
                old_ty,
                None,
                "{:?} has two generic parameters: {:?} and {:?}",
                proj.projection_term,
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
        .chain(folded_requirements.into_iter().map(|clause| Goal::new(cx, param_env, clause)))
        .collect()
}

struct ReplaceProjectionWith<'a, D: SolverDelegate<Interner = I>, I: Interner> {
    ecx: &'a EvalCtxt<'a, D>,
    param_env: I::ParamEnv,
    mapping: HashMap<I::DefId, ty::Binder<I, ty::ProjectionPredicate<I>>>,
    nested: Vec<Goal<I, I::Predicate>>,
}

impl<D: SolverDelegate<Interner = I>, I: Interner> TypeFolder<I>
    for ReplaceProjectionWith<'_, D, I>
{
    fn cx(&self) -> I {
        self.ecx.cx()
    }

    fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
        if let ty::Alias(ty::Projection, alias_ty) = ty.kind() {
            if let Some(replacement) = self.mapping.get(&alias_ty.def_id) {
                // We may have a case where our object type's projection bound is higher-ranked,
                // but the where clauses we instantiated are not. We can solve this by instantiating
                // the binder at the usage site.
                let proj = self.ecx.instantiate_binder_with_infer(*replacement);
                // FIXME: Technically this equate could be fallible...
                self.nested.extend(
                    self.ecx
                        .eq_and_get_goals(
                            self.param_env,
                            alias_ty,
                            proj.projection_term.expect_ty(self.ecx.cx()),
                        )
                        .expect(
                            "expected to be able to unify goal projection with dyn's projection",
                        ),
                );
                proj.term.expect_ty()
            } else {
                ty.super_fold_with(self)
            }
        } else {
            ty.super_fold_with(self)
        }
    }
}
