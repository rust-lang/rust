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
use rustc_macros::{HashStable_NoContext, TyDecodable, TyEncodable};

use crate::inherent::*;
use crate::visit::TypeVisitableExt as _;
use crate::{self as ty, Interner};

/// See `simplify_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
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
        ty::Adt(def, _) => Some(SimplifiedType::Adt(def.def_id())),
        ty::Str => Some(SimplifiedType::Str),
        ty::Array(..) => Some(SimplifiedType::Array),
        ty::Slice(..) => Some(SimplifiedType::Slice),
        ty::Pat(ty, ..) => simplify_type(cx, ty, treat_params),
        ty::RawPtr(_, mutbl) => Some(SimplifiedType::Ptr(mutbl)),
        ty::Dynamic(trait_info, ..) => match trait_info.principal_def_id() {
            Some(principal_def_id) if !cx.trait_is_auto(principal_def_id) => {
                Some(SimplifiedType::Trait(principal_def_id))
            }
            _ => Some(SimplifiedType::MarkerTraitObject),
        },
        ty::Ref(_, _, mutbl) => Some(SimplifiedType::Ref(mutbl)),
        ty::FnDef(def_id, _) | ty::Closure(def_id, _) | ty::CoroutineClosure(def_id, _) => {
            Some(SimplifiedType::Closure(def_id))
        }
        ty::Coroutine(def_id, _) => Some(SimplifiedType::Coroutine(def_id)),
        ty::CoroutineWitness(def_id, _) => Some(SimplifiedType::CoroutineWitness(def_id)),
        ty::Never => Some(SimplifiedType::Never),
        ty::Tuple(tys) => Some(SimplifiedType::Tuple(tys.len())),
        ty::FnPtr(f) => Some(SimplifiedType::Function(f.skip_binder().inputs().len())),
        ty::Placeholder(..) => Some(SimplifiedType::Placeholder),
        ty::Param(_) => match treat_params {
            TreatParams::AsRigid => Some(SimplifiedType::Placeholder),
            TreatParams::InstantiateWithInfer => None,
        },
        ty::Alias(..) => match treat_params {
            // When treating `ty::Param` as a placeholder, projections also
            // don't unify with anything else as long as they are fully normalized.
            // FIXME(-Znext-solver): Can remove this `if` and always simplify to `Placeholder`
            // when the new solver is enabled by default.
            TreatParams::AsRigid if !ty.has_non_region_infer() => Some(SimplifiedType::Placeholder),
            TreatParams::AsRigid | TreatParams::InstantiateWithInfer => None,
        },
        ty::Foreign(def_id) => Some(SimplifiedType::Foreign(def_id)),
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
/// This behavior is toggled using the `TreatParams` fields.
///
/// We use this to quickly reject impl/wc candidates without needing
/// to instantiate generic arguments/having to enter a probe.
///
/// We also use this function during coherence. For coherence the
/// impls only have to overlap for some value, so we treat parameters
/// on both sides like inference variables.
#[derive(Debug, Clone, Copy)]
pub struct DeepRejectCtxt<I: Interner> {
    treat_lhs_params: TreatParams,
    treat_rhs_params: TreatParams,
    _interner: PhantomData<I>,
}

impl<I: Interner> DeepRejectCtxt<I> {
    pub fn new(_interner: I, treat_lhs_params: TreatParams, treat_rhs_params: TreatParams) -> Self {
        DeepRejectCtxt { treat_lhs_params, treat_rhs_params, _interner: PhantomData }
    }

    pub fn args_may_unify(
        self,
        obligation_args: I::GenericArgs,
        impl_args: I::GenericArgs,
    ) -> bool {
        iter::zip(obligation_args.iter(), impl_args.iter()).all(|(obl, imp)| {
            match (obl.kind(), imp.kind()) {
                // We don't fast reject based on regions.
                (ty::GenericArgKind::Lifetime(_), ty::GenericArgKind::Lifetime(_)) => true,
                (ty::GenericArgKind::Type(obl), ty::GenericArgKind::Type(imp)) => {
                    self.types_may_unify(obl, imp)
                }
                (ty::GenericArgKind::Const(obl), ty::GenericArgKind::Const(imp)) => {
                    self.consts_may_unify(obl, imp)
                }
                _ => panic!("kind mismatch: {obl:?} {imp:?}"),
            }
        })
    }

    pub fn types_may_unify(self, lhs: I::Ty, rhs: I::Ty) -> bool {
        match (lhs.kind(), rhs.kind()) {
            (ty::Error(_), _) | (_, ty::Error(_)) => true,

            // As we're walking the whole type, it may encounter projections
            // inside of binders and what not, so we're just going to assume that
            // projections can unify with other stuff.
            //
            // Looking forward to lazy normalization this is the safer strategy anyways.
            (ty::Alias(..), _) | (_, ty::Alias(..)) => true,

            // Bound type variables may unify with rigid types e.g. when using
            // non-lifetime binders.
            (ty::Bound(..), _) | (_, ty::Bound(..)) => true,

            (ty::Infer(var), _) => self.var_and_ty_may_unify(var, rhs),
            (_, ty::Infer(var)) => self.var_and_ty_may_unify(var, lhs),

            (ty::Param(lhs), ty::Param(rhs)) => {
                match (self.treat_lhs_params, self.treat_rhs_params) {
                    (TreatParams::AsRigid, TreatParams::AsRigid) => lhs == rhs,
                    (TreatParams::InstantiateWithInfer, TreatParams::AsRigid)
                    | (TreatParams::AsRigid, TreatParams::InstantiateWithInfer)
                    | (TreatParams::InstantiateWithInfer, TreatParams::InstantiateWithInfer) => {
                        true
                    }
                }
            }
            (ty::Param(_), ty::Placeholder(_)) | (ty::Placeholder(_), ty::Param(_)) => true,
            (ty::Param(_), _) => self.treat_lhs_params == TreatParams::InstantiateWithInfer,
            (_, ty::Param(_)) => self.treat_rhs_params == TreatParams::InstantiateWithInfer,

            // Placeholder types don't unify with anything on their own.
            (ty::Placeholder(lhs), ty::Placeholder(rhs)) => lhs == rhs,
            (ty::Placeholder(_), _) | (_, ty::Placeholder(_)) => false,

            // Purely rigid types, use structural equivalence.
            (ty::Bool, ty::Bool) => lhs == rhs,
            (ty::Bool, _) | (_, ty::Bool) => false,

            (ty::Char, ty::Char) => lhs == rhs,
            (ty::Char, _) | (_, ty::Char) => false,

            (ty::Int(_), ty::Int(_)) => lhs == rhs,
            (ty::Int(_), _) | (_, ty::Int(_)) => false,

            (ty::Uint(_), ty::Uint(_)) => lhs == rhs,
            (ty::Uint(_), _) | (_, ty::Uint(_)) => false,

            (ty::Float(_), ty::Float(_)) => lhs == rhs,
            (ty::Float(_), _) | (_, ty::Float(_)) => false,

            (ty::Str, ty::Str) => lhs == rhs,
            (ty::Str, _) | (_, ty::Str) => false,

            (ty::Never, ty::Never) => lhs == rhs,
            (ty::Never, _) | (_, ty::Never) => false,

            (ty::Foreign(_), ty::Foreign(_)) => lhs == rhs,
            (ty::Foreign(_), _) | (_, ty::Foreign(_)) => false,

            (ty::Ref(_, lhs_ty, lhs_mutbl), ty::Ref(_, rhs_ty, rhs_mutbl)) => {
                lhs_mutbl == rhs_mutbl && self.types_may_unify(lhs_ty, rhs_ty)
            }
            (ty::Ref(..), _) | (_, ty::Ref(..)) => false,

            (ty::Adt(lhs_def, lhs_args), ty::Adt(rhs_def, rhs_args)) => {
                lhs_def == rhs_def && self.args_may_unify(lhs_args, rhs_args)
            }
            (ty::Adt(..), _) | (_, ty::Adt(..)) => false,

            (ty::Pat(lhs_ty, _), ty::Pat(rhs_ty, _)) => {
                // FIXME(pattern_types): take pattern into account
                self.types_may_unify(lhs_ty, rhs_ty)
            }
            (ty::Pat(..), _) | (_, ty::Pat(..)) => false,

            (ty::Slice(lhs_ty), ty::Slice(rhs_ty)) => self.types_may_unify(lhs_ty, rhs_ty),
            (ty::Slice(_), _) | (_, ty::Slice(_)) => false,

            (ty::Array(lhs_ty, lhs_len), ty::Array(rhs_ty, rhs_len)) => {
                self.types_may_unify(lhs_ty, rhs_ty) && self.consts_may_unify(lhs_len, rhs_len)
            }
            (ty::Array(..), _) | (_, ty::Array(..)) => false,

            (ty::Tuple(lhs), ty::Tuple(rhs)) => {
                lhs.len() == rhs.len()
                    && iter::zip(lhs.iter(), rhs.iter())
                        .all(|(lhs, rhs)| self.types_may_unify(lhs, rhs))
            }
            (ty::Tuple(_), _) | (_, ty::Tuple(_)) => false,

            (ty::RawPtr(lhs_ty, lhs_mutbl), ty::RawPtr(rhs_ty, rhs_mutbl)) => {
                lhs_mutbl == rhs_mutbl && self.types_may_unify(lhs_ty, rhs_ty)
            }
            (ty::RawPtr(..), _) | (_, ty::RawPtr(..)) => false,

            (ty::Dynamic(lhs_preds, ..), ty::Dynamic(rhs_preds, ..)) => {
                // Ideally we would walk the existential predicates here or at least
                // compare their length. But considering that the relevant `Relate` impl
                // actually sorts and deduplicates these, that doesn't work.
                lhs_preds.principal_def_id() == rhs_preds.principal_def_id()
            }
            (ty::Dynamic(..), _) | (_, ty::Dynamic(..)) => false,

            (ty::FnPtr(lhs_sig), ty::FnPtr(rhs_sig)) => {
                let lhs_sig = lhs_sig.skip_binder();
                let rhs_sig = rhs_sig.skip_binder();

                lhs_sig.abi == rhs_sig.abi
                    && lhs_sig.c_variadic == rhs_sig.c_variadic
                    && lhs_sig.safety == rhs_sig.safety
                    && lhs_sig.inputs_and_output.len() == rhs_sig.inputs_and_output.len()
                    && iter::zip(lhs_sig.inputs_and_output.iter(), rhs_sig.inputs_and_output.iter())
                        .all(|(lhs, rhs)| self.types_may_unify(lhs, rhs))
            }
            (ty::FnPtr(..), _) | (_, ty::FnPtr(..)) => false,

            (ty::FnDef(lhs_def_id, lhs_args), ty::FnDef(rhs_def_id, rhs_args)) => {
                lhs_def_id == rhs_def_id && self.args_may_unify(lhs_args, rhs_args)
            }
            (ty::FnDef(..), _) | (_, ty::FnDef(..)) => false,

            (ty::Closure(lhs_def_id, lhs_args), ty::Closure(rhs_def_id, rhs_args)) => {
                lhs_def_id == rhs_def_id && self.args_may_unify(lhs_args, rhs_args)
            }
            (ty::Closure(..), _) | (_, ty::Closure(..)) => false,

            (
                ty::CoroutineClosure(lhs_def_id, lhs_args),
                ty::CoroutineClosure(rhs_def_id, rhs_args),
            ) => lhs_def_id == rhs_def_id && self.args_may_unify(lhs_args, rhs_args),
            (ty::CoroutineClosure(..), _) | (_, ty::CoroutineClosure(..)) => false,

            (ty::Coroutine(lhs_def_id, lhs_args), ty::Coroutine(rhs_def_id, rhs_args)) => {
                lhs_def_id == rhs_def_id && self.args_may_unify(lhs_args, rhs_args)
            }
            (ty::Coroutine(..), _) | (_, ty::Coroutine(..)) => false,

            (
                ty::CoroutineWitness(lhs_def_id, lhs_args),
                ty::CoroutineWitness(rhs_def_id, rhs_args),
            ) => lhs_def_id == rhs_def_id && self.args_may_unify(lhs_args, rhs_args),
        }
    }

    pub fn consts_may_unify(self, lhs: I::Const, rhs: I::Const) -> bool {
        // As we don't necessarily eagerly evaluate constants, values
        // may unify with everything except placeholder consts.
        match (lhs.kind(), rhs.kind()) {
            (ty::ConstKind::Value(_, lhs_val), ty::ConstKind::Value(_, rhs_val)) => {
                lhs_val == rhs_val
            }

            (ty::ConstKind::Value(..), ty::ConstKind::Placeholder(_))
            | (ty::ConstKind::Placeholder(_), ty::ConstKind::Value(..)) => false,

            (ty::ConstKind::Param(_), ty::ConstKind::Value(..)) => {
                self.treat_lhs_params == TreatParams::InstantiateWithInfer
            }
            (ty::ConstKind::Value(..), ty::ConstKind::Param(_)) => {
                self.treat_rhs_params == TreatParams::InstantiateWithInfer
            }

            _ => true,
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
