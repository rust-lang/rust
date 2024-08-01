use std::fmt::Debug;
use std::ops::ControlFlow;

use derive_where::derive_where;
use rustc_type_ir::inherent::*;
use rustc_type_ir::visit::{TypeVisitable, TypeVisitableExt, TypeVisitor};
use rustc_type_ir::{self as ty, InferCtxtLike, Interner};
use tracing::instrument;

/// Whether we do the orphan check relative to this crate or to some remote crate.
#[derive(Copy, Clone, Debug)]
pub enum InCrate {
    Local { mode: OrphanCheckMode },
    Remote,
}

#[derive(Copy, Clone, Debug)]
pub enum OrphanCheckMode {
    /// Proper orphan check.
    Proper,
    /// Improper orphan check for backward compatibility.
    ///
    /// In this mode, type params inside projections are considered to be covered
    /// even if the projection may normalize to a type that doesn't actually cover
    /// them. This is unsound. See also [#124559] and [#99554].
    ///
    /// [#124559]: https://github.com/rust-lang/rust/issues/124559
    /// [#99554]: https://github.com/rust-lang/rust/issues/99554
    Compat,
}

#[derive(Debug, Copy, Clone)]
pub enum Conflict {
    Upstream,
    Downstream,
}

/// Returns whether all impls which would apply to the `trait_ref`
/// e.g. `Ty: Trait<Arg>` are already known in the local crate.
///
/// This both checks whether any downstream or sibling crates could
/// implement it and whether an upstream crate can add this impl
/// without breaking backwards compatibility.
#[instrument(level = "debug", skip(infcx, lazily_normalize_ty), ret)]
pub fn trait_ref_is_knowable<Infcx, I, E>(
    infcx: &Infcx,
    trait_ref: ty::TraitRef<I>,
    mut lazily_normalize_ty: impl FnMut(I::Ty) -> Result<I::Ty, E>,
) -> Result<Result<(), Conflict>, E>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    E: Debug,
{
    if orphan_check_trait_ref(infcx, trait_ref, InCrate::Remote, &mut lazily_normalize_ty)?.is_ok()
    {
        // A downstream or cousin crate is allowed to implement some
        // generic parameters of this trait-ref.
        return Ok(Err(Conflict::Downstream));
    }

    if trait_ref_is_local_or_fundamental(infcx.cx(), trait_ref) {
        // This is a local or fundamental trait, so future-compatibility
        // is no concern. We know that downstream/cousin crates are not
        // allowed to implement a generic parameter of this trait ref,
        // which means impls could only come from dependencies of this
        // crate, which we already know about.
        return Ok(Ok(()));
    }

    // This is a remote non-fundamental trait, so if another crate
    // can be the "final owner" of the generic parameters of this trait-ref,
    // they are allowed to implement it future-compatibly.
    //
    // However, if we are a final owner, then nobody else can be,
    // and if we are an intermediate owner, then we don't care
    // about future-compatibility, which means that we're OK if
    // we are an owner.
    if orphan_check_trait_ref(
        infcx,
        trait_ref,
        InCrate::Local { mode: OrphanCheckMode::Proper },
        &mut lazily_normalize_ty,
    )?
    .is_ok()
    {
        Ok(Ok(()))
    } else {
        Ok(Err(Conflict::Upstream))
    }
}

pub fn trait_ref_is_local_or_fundamental<I: Interner>(tcx: I, trait_ref: ty::TraitRef<I>) -> bool {
    trait_ref.def_id.is_local() || tcx.trait_is_fundamental(trait_ref.def_id)
}

#[derive(Debug, Copy, Clone)]
pub enum IsFirstInputType {
    No,
    Yes,
}

impl From<bool> for IsFirstInputType {
    fn from(b: bool) -> IsFirstInputType {
        match b {
            false => IsFirstInputType::No,
            true => IsFirstInputType::Yes,
        }
    }
}

#[derive_where(Debug; I: Interner, T: Debug)]
pub enum OrphanCheckErr<I: Interner, T> {
    NonLocalInputType(Vec<(I::Ty, IsFirstInputType)>),
    UncoveredTyParams(UncoveredTyParams<I, T>),
}

#[derive_where(Debug; I: Interner, T: Debug)]
pub struct UncoveredTyParams<I: Interner, T> {
    pub uncovered: T,
    pub local_ty: Option<I::Ty>,
}

/// Checks whether a trait-ref is potentially implementable by a crate.
///
/// The current rule is that a trait-ref orphan checks in a crate C:
///
/// 1. Order the parameters in the trait-ref in generic parameters order
/// - Self first, others linearly (e.g., `<U as Foo<V, W>>` is U < V < W).
/// 2. Of these type parameters, there is at least one type parameter
///    in which, walking the type as a tree, you can reach a type local
///    to C where all types in-between are fundamental types. Call the
///    first such parameter the "local key parameter".
///     - e.g., `Box<LocalType>` is OK, because you can visit LocalType
///       going through `Box`, which is fundamental.
///     - similarly, `FundamentalPair<Vec<()>, Box<LocalType>>` is OK for
///       the same reason.
///     - but (knowing that `Vec<T>` is non-fundamental, and assuming it's
///       not local), `Vec<LocalType>` is bad, because `Vec<->` is between
///       the local type and the type parameter.
/// 3. Before this local type, no generic type parameter of the impl must
///    be reachable through fundamental types.
///     - e.g. `impl<T> Trait<LocalType> for Vec<T>` is fine, as `Vec` is not fundamental.
///     - while `impl<T> Trait<LocalType> for Box<T>` results in an error, as `T` is
///       reachable through the fundamental type `Box`.
/// 4. Every type in the local key parameter not known in C, going
///    through the parameter's type tree, must appear only as a subtree of
///    a type local to C, with only fundamental types between the type
///    local to C and the local key parameter.
///     - e.g., `Vec<LocalType<T>>>` (or equivalently `Box<Vec<LocalType<T>>>`)
///     is bad, because the only local type with `T` as a subtree is
///     `LocalType<T>`, and `Vec<->` is between it and the type parameter.
///     - similarly, `FundamentalPair<LocalType<T>, T>` is bad, because
///     the second occurrence of `T` is not a subtree of *any* local type.
///     - however, `LocalType<Vec<T>>` is OK, because `T` is a subtree of
///     `LocalType<Vec<T>>`, which is local and has no types between it and
///     the type parameter.
///
/// The orphan rules actually serve several different purposes:
///
/// 1. They enable link-safety - i.e., 2 mutually-unknowing crates (where
///    every type local to one crate is unknown in the other) can't implement
///    the same trait-ref. This follows because it can be seen that no such
///    type can orphan-check in 2 such crates.
///
///    To check that a local impl follows the orphan rules, we check it in
///    InCrate::Local mode, using type parameters for the "generic" types.
///
///    In InCrate::Local mode the orphan check succeeds if the current crate
///    is definitely allowed to implement the given trait (no false positives).
///
/// 2. They ground negative reasoning for coherence. If a user wants to
///    write both a conditional blanket impl and a specific impl, we need to
///    make sure they do not overlap. For example, if we write
///    ```ignore (illustrative)
///    impl<T> IntoIterator for Vec<T>
///    impl<T: Iterator> IntoIterator for T
///    ```
///    We need to be able to prove that `Vec<$0>: !Iterator` for every type $0.
///    We can observe that this holds in the current crate, but we need to make
///    sure this will also hold in all unknown crates (both "independent" crates,
///    which we need for link-safety, and also child crates, because we don't want
///    child crates to get error for impl conflicts in a *dependency*).
///
///    For that, we only allow negative reasoning if, for every assignment to the
///    inference variables, every unknown crate would get an orphan error if they
///    try to implement this trait-ref. To check for this, we use InCrate::Remote
///    mode. That is sound because we already know all the impls from known crates.
///
///    In InCrate::Remote mode the orphan check succeeds if a foreign crate
///    *could* implement the given trait (no false negatives).
///
/// 3. For non-`#[fundamental]` traits, they guarantee that parent crates can
///    add "non-blanket" impls without breaking negative reasoning in dependent
///    crates. This is the "rebalancing coherence" (RFC 1023) restriction.
///
///    For that, we only allow a crate to perform negative reasoning on
///    non-local-non-`#[fundamental]` if there's a local key parameter as per (2).
///
///    Because we never perform negative reasoning generically (coherence does
///    not involve type parameters), this can be interpreted as doing the full
///    orphan check (using InCrate::Local mode), instantiating non-local known
///    types for all inference variables.
///
///    This allows for crates to future-compatibly add impls as long as they
///    can't apply to types with a key parameter in a child crate - applying
///    the rules, this basically means that every type parameter in the impl
///    must appear behind a non-fundamental type (because this is not a
///    type-system requirement, crate owners might also go for "semantic
///    future-compatibility" involving things such as sealed traits, but
///    the above requirement is sufficient, and is necessary in "open world"
///    cases).
///
/// Note that this function is never called for types that have both type
/// parameters and inference variables.
#[instrument(level = "trace", skip(infcx, lazily_normalize_ty), ret)]
pub fn orphan_check_trait_ref<Infcx, I, E: Debug>(
    infcx: &Infcx,
    trait_ref: ty::TraitRef<I>,
    in_crate: InCrate,
    lazily_normalize_ty: impl FnMut(I::Ty) -> Result<I::Ty, E>,
) -> Result<Result<(), OrphanCheckErr<I, I::Ty>>, E>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    E: Debug,
{
    if trait_ref.has_param() {
        panic!("orphan check only expects inference variables: {trait_ref:?}");
    }

    let mut checker = OrphanChecker::new(infcx, in_crate, lazily_normalize_ty);
    Ok(match trait_ref.visit_with(&mut checker) {
        ControlFlow::Continue(()) => Err(OrphanCheckErr::NonLocalInputType(checker.non_local_tys)),
        ControlFlow::Break(residual) => match residual {
            OrphanCheckEarlyExit::NormalizationFailure(err) => return Err(err),
            OrphanCheckEarlyExit::UncoveredTyParam(ty) => {
                // Does there exist some local type after the `ParamTy`.
                checker.search_first_local_ty = true;
                let local_ty = match trait_ref.visit_with(&mut checker) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(local_ty)) => Some(local_ty),
                    _ => None,
                };
                Err(OrphanCheckErr::UncoveredTyParams(UncoveredTyParams {
                    uncovered: ty,
                    local_ty,
                }))
            }
            OrphanCheckEarlyExit::LocalTy(_) => Ok(()),
        },
    })
}

struct OrphanChecker<'a, Infcx, I: Interner, F> {
    infcx: &'a Infcx,
    in_crate: InCrate,
    in_self_ty: bool,
    lazily_normalize_ty: F,
    /// Ignore orphan check failures and exclusively search for the first local type.
    search_first_local_ty: bool,
    non_local_tys: Vec<(I::Ty, IsFirstInputType)>,
}

impl<'a, Infcx, I, F, E> OrphanChecker<'a, Infcx, I, F>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    F: FnOnce(I::Ty) -> Result<I::Ty, E>,
{
    fn new(infcx: &'a Infcx, in_crate: InCrate, lazily_normalize_ty: F) -> Self {
        OrphanChecker {
            infcx,
            in_crate,
            in_self_ty: true,
            lazily_normalize_ty,
            search_first_local_ty: false,
            non_local_tys: Vec::new(),
        }
    }

    fn found_non_local_ty(&mut self, t: I::Ty) -> ControlFlow<OrphanCheckEarlyExit<I, E>> {
        self.non_local_tys.push((t, self.in_self_ty.into()));
        ControlFlow::Continue(())
    }

    fn found_uncovered_ty_param(&mut self, ty: I::Ty) -> ControlFlow<OrphanCheckEarlyExit<I, E>> {
        if self.search_first_local_ty {
            return ControlFlow::Continue(());
        }

        ControlFlow::Break(OrphanCheckEarlyExit::UncoveredTyParam(ty))
    }

    fn def_id_is_local(&mut self, def_id: I::DefId) -> bool {
        match self.in_crate {
            InCrate::Local { .. } => def_id.is_local(),
            InCrate::Remote => false,
        }
    }
}

enum OrphanCheckEarlyExit<I: Interner, E> {
    NormalizationFailure(E),
    UncoveredTyParam(I::Ty),
    LocalTy(I::Ty),
}

impl<'a, Infcx, I, F, E> TypeVisitor<I> for OrphanChecker<'a, Infcx, I, F>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    F: FnMut(I::Ty) -> Result<I::Ty, E>,
{
    type Result = ControlFlow<OrphanCheckEarlyExit<I, E>>;

    fn visit_region(&mut self, _r: I::Region) -> Self::Result {
        ControlFlow::Continue(())
    }

    fn visit_ty(&mut self, ty: I::Ty) -> Self::Result {
        let ty = self.infcx.shallow_resolve(ty);
        let ty = match (self.lazily_normalize_ty)(ty) {
            Ok(norm_ty) if norm_ty.is_ty_var() => ty,
            Ok(norm_ty) => norm_ty,
            Err(err) => return ControlFlow::Break(OrphanCheckEarlyExit::NormalizationFailure(err)),
        };

        let result = match ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Str
            | ty::FnDef(..)
            | ty::Pat(..)
            | ty::FnPtr(_)
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Never
            | ty::Tuple(..) => self.found_non_local_ty(ty),

            ty::Param(..) => panic!("unexpected ty param"),

            ty::Placeholder(..) | ty::Bound(..) | ty::Infer(..) => {
                match self.in_crate {
                    InCrate::Local { .. } => self.found_uncovered_ty_param(ty),
                    // The inference variable might be unified with a local
                    // type in that remote crate.
                    InCrate::Remote => ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty)),
                }
            }

            // A rigid alias may normalize to anything.
            // * If it references an infer var, placeholder or bound ty, it may
            //   normalize to that, so we have to treat it as an uncovered ty param.
            // * Otherwise it may normalize to any non-type-generic type
            //   be it local or non-local.
            ty::Alias(kind, _) => {
                if ty.has_type_flags(
                    ty::TypeFlags::HAS_TY_PLACEHOLDER
                        | ty::TypeFlags::HAS_TY_BOUND
                        | ty::TypeFlags::HAS_TY_INFER,
                ) {
                    match self.in_crate {
                        InCrate::Local { mode } => match kind {
                            ty::Projection => {
                                if let OrphanCheckMode::Compat = mode {
                                    ControlFlow::Continue(())
                                } else {
                                    self.found_uncovered_ty_param(ty)
                                }
                            }
                            _ => self.found_uncovered_ty_param(ty),
                        },
                        InCrate::Remote => {
                            // The inference variable might be unified with a local
                            // type in that remote crate.
                            ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                        }
                    }
                } else {
                    // Regarding *opaque types* specifically, we choose to treat them as non-local,
                    // even those that appear within the same crate. This seems somewhat surprising
                    // at first, but makes sense when you consider that opaque types are supposed
                    // to hide the underlying type *within the same crate*. When an opaque type is
                    // used from outside the module where it is declared, it should be impossible to
                    // observe anything about it other than the traits that it implements.
                    //
                    // The alternative would be to look at the underlying type to determine whether
                    // or not the opaque type itself should be considered local.
                    //
                    // However, this could make it a breaking change to switch the underlying hidden
                    // type from a local type to a remote type. This would violate the rule that
                    // opaque types should be completely opaque apart from the traits that they
                    // implement, so we don't use this behavior.
                    // Addendum: Moreover, revealing the underlying type is likely to cause cycle
                    // errors as we rely on coherence / the specialization graph during typeck.

                    self.found_non_local_ty(ty)
                }
            }

            // For fundamental types, we just look inside of them.
            ty::Ref(_, ty, _) => ty.visit_with(self),
            ty::Adt(def, args) => {
                if self.def_id_is_local(def.def_id()) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                } else if def.is_fundamental() {
                    args.visit_with(self)
                } else {
                    self.found_non_local_ty(ty)
                }
            }
            ty::Foreign(def_id) => {
                if self.def_id_is_local(def_id) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                } else {
                    self.found_non_local_ty(ty)
                }
            }
            ty::Dynamic(tt, ..) => {
                let principal = tt.principal().map(|p| p.def_id());
                if principal.is_some_and(|p| self.def_id_is_local(p)) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                } else {
                    self.found_non_local_ty(ty)
                }
            }
            ty::Error(_) => ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty)),
            ty::Closure(did, ..) | ty::CoroutineClosure(did, ..) | ty::Coroutine(did, ..) => {
                if self.def_id_is_local(did) {
                    ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty))
                } else {
                    self.found_non_local_ty(ty)
                }
            }
            // This should only be created when checking whether we have to check whether some
            // auto trait impl applies. There will never be multiple impls, so we can just
            // act as if it were a local type here.
            ty::CoroutineWitness(..) => ControlFlow::Break(OrphanCheckEarlyExit::LocalTy(ty)),
        };
        // A bit of a hack, the `OrphanChecker` is only used to visit a `TraitRef`, so
        // the first type we visit is always the self type.
        self.in_self_ty = false;
        result
    }

    /// All possible values for a constant parameter already exist
    /// in the crate defining the trait, so they are always non-local[^1].
    ///
    /// Because there's no way to have an impl where the first local
    /// generic argument is a constant, we also don't have to fail
    /// the orphan check when encountering a parameter or a generic constant.
    ///
    /// This means that we can completely ignore constants during the orphan check.
    ///
    /// See `tests/ui/coherence/const-generics-orphan-check-ok.rs` for examples.
    ///
    /// [^1]: This might not hold for function pointers or trait objects in the future.
    /// As these should be quite rare as const arguments and especially rare as impl
    /// parameters, allowing uncovered const parameters in impls seems more useful
    /// than allowing `impl<T> Trait<local_fn_ptr, T> for i32` to compile.
    fn visit_const(&mut self, _c: I::Const) -> Self::Result {
        ControlFlow::Continue(())
    }
}
