use std::fmt::Debug;
use std::hash::Hash;
use std::iter;
use std::marker::PhantomData;

use rustc_ast_ir::Mutability;
#[cfg(feature = "nightly")]
use rustc_data_structures::fingerprint::Fingerprint;
#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::{HashStable, StableHasher, ToStableHashKey};
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};

use crate::inherent::*;
use crate::visit::TypeVisitableExt as _;
use crate::{self as ty, Interner};

/// See `simplify_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub enum SimplifiedType<DefId> {
    Bool,
    Char,
    Int(ty::IntTy),
    Uint(ty::UintTy),
    Float(ty::FloatTy),
    Adt(DefId),
    Foreign(DefId),
    Str,
    Array,
    Slice,
    Ref(Mutability),
    Ptr(Mutability),
    Never,
    Tuple(usize),
    /// A trait object, all of whose components are markers
    /// (e.g., `dyn Send + Sync`).
    MarkerTraitObject,
    Trait(DefId),
    Closure(DefId),
    Coroutine(DefId),
    CoroutineWitness(DefId),
    Function(usize),
    UnsafeBinder,
    Placeholder,
    Error,
}

#[cfg(feature = "nightly")]
impl<HCX: Clone, DefId: HashStable<HCX>> ToStableHashKey<HCX> for SimplifiedType<DefId> {
    type KeyType = Fingerprint;

    #[inline]
    fn to_stable_hash_key(&self, hcx: &HCX) -> Fingerprint {
        let mut hasher = StableHasher::new();
        let mut hcx: HCX = hcx.clone();
        self.hash_stable(&mut hcx, &mut hasher);
        hasher.finish()
    }
}

/// Generic parameters are pretty much just bound variables, e.g.
/// the type of `fn foo<'a, T>(x: &'a T) -> u32 { ... }` can be thought of as
/// `for<'a, T> fn(&'a T) -> u32`.
///
/// Typecheck of `foo` has to succeed for all possible generic arguments, so
/// during typeck, we have to treat its generic parameters as if they
/// were placeholders.
///
/// But when calling `foo` we only have to provide a specific generic argument.
/// In that case the generic parameters are instantiated with inference variables.
/// As we use `simplify_type` before that instantiation happens, we just treat
/// generic parameters as if they were inference variables in that case.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum TreatParams {
    /// Treat parameters as infer vars. This is the correct mode for caching
    /// an impl's type for lookup.
    InstantiateWithInfer,
    /// Treat parameters as placeholders in the given environment. This is the
    /// correct mode for *lookup*, as during candidate selection.
    ///
    /// This also treats projections with inference variables as infer vars
    /// since they could be further normalized.
    // FIXME(@lcnr): This treats aliases as rigid. This is only correct if the
    // type has been structurally normalized. We should reflect this requirement
    // in the variant name. It is currently incorrectly used in diagnostics.
    AsRigid,
}

/// Tries to simplify a type by only returning the outermost injective¹ layer, if one exists.
///
/// **This function should only be used if you need to store or retrieve the type from some
/// hashmap. If you want to quickly decide whether two types may unify, use the [DeepRejectCtxt]
/// instead.**
///
/// The idea is to get something simple that we can use to quickly decide if two types could unify,
/// for example during method lookup. If this function returns `Some(x)` it can only unify with
/// types for which this method returns either `Some(x)` as well or `None`.
///
/// A special case here are parameters and projections, which are only injective
/// if they are treated as placeholders.
///
/// For example when storing impls based on their simplified self type, we treat
/// generic parameters as if they were inference variables. We must not simplify them here,
/// as they can unify with any other type.
///
/// With projections we have to be even more careful, as treating them as placeholders
/// is only correct if they are fully normalized.
///
/// ¹ meaning that if the outermost layers are different, then the whole types are also different.
pub fn simplify_type<I: Interner>(
    cx: I,
    ty: I::Ty,
    treat_params: TreatParams,
) -> Option<SimplifiedType<I::DefId>> {
    match ty.kind() {
        ty::Bool => Some(SimplifiedType::Bool),
        ty::Char => Some(SimplifiedType::Char),
        ty::Int(int_type) => Some(SimplifiedType::Int(int_type)),
        ty::Uint(uint_type) => Some(SimplifiedType::Uint(uint_type)),
        ty::Float(float_type) => Some(SimplifiedType::Float(float_type)),
        ty::Adt(def, _) => Some(SimplifiedType::Adt(def.def_id().into())),
        ty::Str => Some(SimplifiedType::Str),
        ty::Array(..) => Some(SimplifiedType::Array),
        ty::Slice(..) => Some(SimplifiedType::Slice),
        ty::Pat(ty, ..) => simplify_type(cx, ty, treat_params),
        ty::RawPtr(_, mutbl) => Some(SimplifiedType::Ptr(mutbl)),
        ty::Dynamic(trait_info, ..) => match trait_info.principal_def_id() {
            Some(principal_def_id) if !cx.trait_is_auto(principal_def_id) => {
                Some(SimplifiedType::Trait(principal_def_id.into()))
            }
            _ => Some(SimplifiedType::MarkerTraitObject),
        },
        ty::Ref(_, _, mutbl) => Some(SimplifiedType::Ref(mutbl)),
        ty::FnDef(def_id, _) => Some(SimplifiedType::Closure(def_id.into())),
        ty::Closure(def_id, _) => Some(SimplifiedType::Closure(def_id.into())),
        ty::CoroutineClosure(def_id, _) => Some(SimplifiedType::Closure(def_id.into())),
        ty::Coroutine(def_id, _) => Some(SimplifiedType::Coroutine(def_id.into())),
        ty::CoroutineWitness(def_id, _) => Some(SimplifiedType::CoroutineWitness(def_id.into())),
        ty::Never => Some(SimplifiedType::Never),
        ty::Tuple(tys) => Some(SimplifiedType::Tuple(tys.len())),
        ty::FnPtr(sig_tys, _hdr) => {
            Some(SimplifiedType::Function(sig_tys.skip_binder().inputs().len()))
        }
        ty::UnsafeBinder(_) => Some(SimplifiedType::UnsafeBinder),
        ty::Placeholder(..) => Some(SimplifiedType::Placeholder),
        ty::Param(_) => match treat_params {
            TreatParams::AsRigid => Some(SimplifiedType::Placeholder),
            TreatParams::InstantiateWithInfer => None,
        },
        ty::Alias(..) => match treat_params {
            // When treating `ty::Param` as a placeholder, projections also
            // don't unify with anything else as long as they are fully normalized.
            TreatParams::AsRigid
                if !ty.has_non_region_infer() || cx.next_trait_solver_globally() =>
            {
                Some(SimplifiedType::Placeholder)
            }
            TreatParams::AsRigid | TreatParams::InstantiateWithInfer => None,
        },
        ty::Foreign(def_id) => Some(SimplifiedType::Foreign(def_id.into())),
        ty::Error(_) => Some(SimplifiedType::Error),
        ty::Bound(..) | ty::Infer(_) => None,
    }
}

impl<DefId> SimplifiedType<DefId> {
    pub fn def(self) -> Option<DefId> {
        match self {
            SimplifiedType::Adt(d)
            | SimplifiedType::Foreign(d)
            | SimplifiedType::Trait(d)
            | SimplifiedType::Closure(d)
            | SimplifiedType::Coroutine(d)
            | SimplifiedType::CoroutineWitness(d) => Some(d),
            _ => None,
        }
    }
}

/// Given generic arguments, could they be unified after
/// replacing parameters with inference variables or placeholders.
/// This behavior is toggled using the const generics.
///
/// We use this to quickly reject impl/wc candidates without needing
/// to instantiate generic arguments/having to enter a probe.
///
/// We also use this function during coherence. For coherence the
/// impls only have to overlap for some value, so we treat parameters
/// on both sides like inference variables.
#[derive(Debug, Clone, Copy)]
pub struct DeepRejectCtxt<
    I: Interner,
    const INSTANTIATE_LHS_WITH_INFER: bool,
    const INSTANTIATE_RHS_WITH_INFER: bool,
> {
    _interner: PhantomData<I>,
}

impl<I: Interner> DeepRejectCtxt<I, false, false> {
    /// Treat parameters in both the lhs and the rhs as rigid.
    pub fn relate_rigid_rigid(_interner: I) -> DeepRejectCtxt<I, false, false> {
        DeepRejectCtxt { _interner: PhantomData }
    }
}

impl<I: Interner> DeepRejectCtxt<I, true, true> {
    /// Treat parameters in both the lhs and the rhs as infer vars.
    pub fn relate_infer_infer(_interner: I) -> DeepRejectCtxt<I, true, true> {
        DeepRejectCtxt { _interner: PhantomData }
    }
}

impl<I: Interner> DeepRejectCtxt<I, false, true> {
    /// Treat parameters in the lhs as rigid, and in rhs as infer vars.
    pub fn relate_rigid_infer(_interner: I) -> DeepRejectCtxt<I, false, true> {
        DeepRejectCtxt { _interner: PhantomData }
    }
}

impl<I: Interner, const INSTANTIATE_LHS_WITH_INFER: bool, const INSTANTIATE_RHS_WITH_INFER: bool>
    DeepRejectCtxt<I, INSTANTIATE_LHS_WITH_INFER, INSTANTIATE_RHS_WITH_INFER>
{
    // Quite arbitrary. Large enough to only affect a very tiny amount of impls/crates
    // and small enough to prevent hangs.
    const STARTING_DEPTH: usize = 8;

    pub fn args_may_unify(
        self,
        obligation_args: I::GenericArgs,
        impl_args: I::GenericArgs,
    ) -> bool {
        self.args_may_unify_inner(obligation_args, impl_args, Self::STARTING_DEPTH)
    }

    pub fn types_may_unify(self, lhs: I::Ty, rhs: I::Ty) -> bool {
        self.types_may_unify_inner(lhs, rhs, Self::STARTING_DEPTH)
    }

    pub fn types_may_unify_with_depth(self, lhs: I::Ty, rhs: I::Ty, depth_limit: usize) -> bool {
        self.types_may_unify_inner(lhs, rhs, depth_limit)
    }

    fn args_may_unify_inner(
        self,
        obligation_args: I::GenericArgs,
        impl_args: I::GenericArgs,
        depth: usize,
    ) -> bool {
        // No need to decrement the depth here as this function is only
        // recursively reachable via `types_may_unify_inner` which already
        // increments the depth for us.
        iter::zip(obligation_args.iter(), impl_args.iter()).all(|(obl, imp)| {
            match (obl.kind(), imp.kind()) {
                // We don't fast reject based on regions.
                (ty::GenericArgKind::Lifetime(_), ty::GenericArgKind::Lifetime(_)) => true,
                (ty::GenericArgKind::Type(obl), ty::GenericArgKind::Type(imp)) => {
                    self.types_may_unify_inner(obl, imp, depth)
                }
                (ty::GenericArgKind::Const(obl), ty::GenericArgKind::Const(imp)) => {
                    self.consts_may_unify_inner(obl, imp)
                }
                _ => panic!("kind mismatch: {obl:?} {imp:?}"),
            }
        })
    }

    fn types_may_unify_inner(self, lhs: I::Ty, rhs: I::Ty, depth: usize) -> bool {
        if lhs == rhs {
            return true;
        }

        match rhs.kind() {
            // Start by checking whether the `rhs` type may unify with
            // pretty much everything. Just return `true` in that case.
            ty::Param(_) => {
                if INSTANTIATE_RHS_WITH_INFER {
                    return true;
                }
            }
            ty::Error(_) | ty::Alias(..) | ty::Bound(..) => return true,
            ty::Infer(var) => return self.var_and_ty_may_unify(var, lhs),

            // These types only unify with inference variables or their own
            // variant.
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(..)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Dynamic(..)
            | ty::Pat(..)
            | ty::Ref(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Foreign(_)
            | ty::Placeholder(_)
            | ty::UnsafeBinder(_) => {}
        };

        // The type system needs to support exponentially large types
        // as long as they are self-similar. While most other folders
        // use caching to handle them, this folder exists purely as a
        // perf optimization and is incredibly hot. In pretty much all
        // uses checking the cache is slower than simply recursing, so
        // we instead just add an arbitrary depth cutoff.
        //
        // We only decrement the depth here as the match on `rhs`
        // does not recurse.
        let Some(depth) = depth.checked_sub(1) else {
            return true;
        };

        // For purely rigid types, use structural equivalence.
        match lhs.kind() {
            ty::Ref(_, lhs_ty, lhs_mutbl) => match rhs.kind() {
                ty::Ref(_, rhs_ty, rhs_mutbl) => {
                    lhs_mutbl == rhs_mutbl && self.types_may_unify_inner(lhs_ty, rhs_ty, depth)
                }
                _ => false,
            },

            ty::Adt(lhs_def, lhs_args) => match rhs.kind() {
                ty::Adt(rhs_def, rhs_args) => {
                    lhs_def == rhs_def && self.args_may_unify_inner(lhs_args, rhs_args, depth)
                }
                _ => false,
            },

            // Depending on the value of const generics, we either treat generic parameters
            // like placeholders or like inference variables.
            ty::Param(lhs) => {
                INSTANTIATE_LHS_WITH_INFER
                    || match rhs.kind() {
                        ty::Param(rhs) => lhs == rhs,
                        _ => false,
                    }
            }

            // Placeholder types don't unify with anything on their own.
            ty::Placeholder(lhs) => {
                matches!(rhs.kind(), ty::Placeholder(rhs) if lhs == rhs)
            }

            ty::Infer(var) => self.var_and_ty_may_unify(var, rhs),

            // As we're walking the whole type, it may encounter projections
            // inside of binders and what not, so we're just going to assume that
            // projections can unify with other stuff.
            //
            // Looking forward to lazy normalization this is the safer strategy anyways.
            ty::Alias(..) => true,

            ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Bool
            | ty::Char
            | ty::Never
            | ty::Foreign(_) => lhs == rhs,

            ty::Tuple(lhs) => match rhs.kind() {
                ty::Tuple(rhs) => {
                    lhs.len() == rhs.len()
                        && iter::zip(lhs.iter(), rhs.iter())
                            .all(|(lhs, rhs)| self.types_may_unify_inner(lhs, rhs, depth))
                }
                _ => false,
            },

            ty::Array(lhs_ty, lhs_len) => match rhs.kind() {
                ty::Array(rhs_ty, rhs_len) => {
                    self.types_may_unify_inner(lhs_ty, rhs_ty, depth)
                        && self.consts_may_unify_inner(lhs_len, rhs_len)
                }
                _ => false,
            },

            ty::RawPtr(lhs_ty, lhs_mutbl) => match rhs.kind() {
                ty::RawPtr(rhs_ty, rhs_mutbl) => {
                    lhs_mutbl == rhs_mutbl && self.types_may_unify_inner(lhs_ty, rhs_ty, depth)
                }
                _ => false,
            },

            ty::Slice(lhs_ty) => {
                matches!(rhs.kind(), ty::Slice(rhs_ty) if self.types_may_unify_inner(lhs_ty, rhs_ty, depth))
            }

            ty::Dynamic(lhs_preds, ..) => {
                // Ideally we would walk the existential predicates here or at least
                // compare their length. But considering that the relevant `Relate` impl
                // actually sorts and deduplicates these, that doesn't work.
                matches!(rhs.kind(), ty::Dynamic(rhs_preds, ..) if
                    lhs_preds.principal_def_id() == rhs_preds.principal_def_id()
                )
            }

            ty::FnPtr(lhs_sig_tys, lhs_hdr) => match rhs.kind() {
                ty::FnPtr(rhs_sig_tys, rhs_hdr) => {
                    let lhs_sig_tys = lhs_sig_tys.skip_binder().inputs_and_output;
                    let rhs_sig_tys = rhs_sig_tys.skip_binder().inputs_and_output;

                    lhs_hdr == rhs_hdr
                        && lhs_sig_tys.len() == rhs_sig_tys.len()
                        && iter::zip(lhs_sig_tys.iter(), rhs_sig_tys.iter())
                            .all(|(lhs, rhs)| self.types_may_unify_inner(lhs, rhs, depth))
                }
                _ => false,
            },

            ty::Bound(..) => true,

            ty::FnDef(lhs_def_id, lhs_args) => match rhs.kind() {
                ty::FnDef(rhs_def_id, rhs_args) => {
                    lhs_def_id == rhs_def_id && self.args_may_unify_inner(lhs_args, rhs_args, depth)
                }
                _ => false,
            },

            ty::Closure(lhs_def_id, lhs_args) => match rhs.kind() {
                ty::Closure(rhs_def_id, rhs_args) => {
                    lhs_def_id == rhs_def_id && self.args_may_unify_inner(lhs_args, rhs_args, depth)
                }
                _ => false,
            },

            ty::CoroutineClosure(lhs_def_id, lhs_args) => match rhs.kind() {
                ty::CoroutineClosure(rhs_def_id, rhs_args) => {
                    lhs_def_id == rhs_def_id && self.args_may_unify_inner(lhs_args, rhs_args, depth)
                }
                _ => false,
            },

            ty::Coroutine(lhs_def_id, lhs_args) => match rhs.kind() {
                ty::Coroutine(rhs_def_id, rhs_args) => {
                    lhs_def_id == rhs_def_id && self.args_may_unify_inner(lhs_args, rhs_args, depth)
                }
                _ => false,
            },

            ty::CoroutineWitness(lhs_def_id, lhs_args) => match rhs.kind() {
                ty::CoroutineWitness(rhs_def_id, rhs_args) => {
                    lhs_def_id == rhs_def_id && self.args_may_unify_inner(lhs_args, rhs_args, depth)
                }
                _ => false,
            },

            ty::Pat(lhs_ty, _) => {
                // FIXME(pattern_types): take pattern into account
                matches!(rhs.kind(), ty::Pat(rhs_ty, _) if self.types_may_unify_inner(lhs_ty, rhs_ty, depth))
            }

            ty::UnsafeBinder(lhs_ty) => match rhs.kind() {
                ty::UnsafeBinder(rhs_ty) => {
                    self.types_may_unify(lhs_ty.skip_binder(), rhs_ty.skip_binder())
                }
                _ => false,
            },

            ty::Error(..) => true,
        }
    }

    // Unlike `types_may_unify_inner`, this does not take a depth as
    // we never recurse from this function.
    fn consts_may_unify_inner(self, lhs: I::Const, rhs: I::Const) -> bool {
        match rhs.kind() {
            ty::ConstKind::Param(_) => {
                if INSTANTIATE_RHS_WITH_INFER {
                    return true;
                }
            }

            ty::ConstKind::Expr(_)
            | ty::ConstKind::Unevaluated(_)
            | ty::ConstKind::Error(_)
            | ty::ConstKind::Infer(_)
            | ty::ConstKind::Bound(..) => {
                return true;
            }

            ty::ConstKind::Value(..) | ty::ConstKind::Placeholder(_) => {}
        };

        match lhs.kind() {
            ty::ConstKind::Value(lhs_val) => match rhs.kind() {
                ty::ConstKind::Value(rhs_val) => lhs_val.valtree() == rhs_val.valtree(),
                _ => false,
            },

            ty::ConstKind::Param(lhs) => {
                INSTANTIATE_LHS_WITH_INFER
                    || match rhs.kind() {
                        ty::ConstKind::Param(rhs) => lhs == rhs,
                        _ => false,
                    }
            }

            // Placeholder consts don't unify with anything on their own
            ty::ConstKind::Placeholder(lhs) => {
                matches!(rhs.kind(), ty::ConstKind::Placeholder(rhs) if lhs == rhs)
            }

            // As we don't necessarily eagerly evaluate constants,
            // they might unify with any value.
            ty::ConstKind::Expr(_) | ty::ConstKind::Unevaluated(_) | ty::ConstKind::Error(_) => {
                true
            }

            ty::ConstKind::Infer(_) | ty::ConstKind::Bound(..) => true,
        }
    }

    fn var_and_ty_may_unify(self, var: ty::InferTy, ty: I::Ty) -> bool {
        if !ty.is_known_rigid() {
            return true;
        }

        match var {
            ty::IntVar(_) => ty.is_integral(),
            ty::FloatVar(_) => ty.is_floating_point(),
            _ => true,
        }
    }
}
