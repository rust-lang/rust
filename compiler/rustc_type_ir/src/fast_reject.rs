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
    AsCandidateKey,
    /// Treat parameters as placeholders in the given environment. This is the
    /// correct mode for *lookup*, as during candidate selection.
    ///
    /// This also treats projections with inference variables as infer vars
    /// since they could be further normalized.
    ForLookup,
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
    tcx: I,
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
        ty::Pat(ty, ..) => simplify_type(tcx, ty, treat_params),
        ty::RawPtr(_, mutbl) => Some(SimplifiedType::Ptr(mutbl)),
        ty::Dynamic(trait_info, ..) => match trait_info.principal_def_id() {
            Some(principal_def_id) if !tcx.trait_is_auto(principal_def_id) => {
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
            TreatParams::ForLookup => Some(SimplifiedType::Placeholder),
            TreatParams::AsCandidateKey => None,
        },
        ty::Alias(..) => match treat_params {
            // When treating `ty::Param` as a placeholder, projections also
            // don't unify with anything else as long as they are fully normalized.
            // FIXME(-Znext-solver): Can remove this `if` and always simplify to `Placeholder`
            // when the new solver is enabled by default.
            TreatParams::ForLookup if !ty.has_non_region_infer() => {
                Some(SimplifiedType::Placeholder)
            }
            TreatParams::ForLookup | TreatParams::AsCandidateKey => None,
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

/// Given generic arguments from an obligation and an impl,
/// could these two be unified after replacing parameters in the
/// the impl with inference variables.
///
/// For obligations, parameters won't be replaced by inference
/// variables and only unify with themselves. We treat them
/// the same way we treat placeholders.
///
/// We also use this function during coherence. For coherence the
/// impls only have to overlap for some value, so we treat parameters
/// on both sides like inference variables. This behavior is toggled
/// using the `treat_obligation_params` field.
#[derive(Debug, Clone, Copy)]
pub struct DeepRejectCtxt<I: Interner> {
    treat_obligation_params: TreatParams,
    _interner: PhantomData<I>,
}

impl<I: Interner> DeepRejectCtxt<I> {
    pub fn new(_interner: I, treat_obligation_params: TreatParams) -> Self {
        DeepRejectCtxt { treat_obligation_params, _interner: PhantomData }
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

    pub fn types_may_unify(self, obligation_ty: I::Ty, impl_ty: I::Ty) -> bool {
        match impl_ty.kind() {
            // Start by checking whether the type in the impl may unify with
            // pretty much everything. Just return `true` in that case.
            ty::Param(_) | ty::Error(_) | ty::Alias(..) => return true,
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
            | ty::FnPtr(..)
            | ty::Foreign(..) => debug_assert!(impl_ty.is_known_rigid()),
            ty::FnDef(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Infer(_) => panic!("unexpected impl_ty: {impl_ty:?}"),
        }

        let k = impl_ty.kind();
        match obligation_ty.kind() {
            // Purely rigid types, use structural equivalence.
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Never
            | ty::Foreign(_) => obligation_ty == impl_ty,
            ty::Ref(_, obl_ty, obl_mutbl) => match k {
                ty::Ref(_, impl_ty, impl_mutbl) => {
                    obl_mutbl == impl_mutbl && self.types_may_unify(obl_ty, impl_ty)
                }
                _ => false,
            },
            ty::Adt(obl_def, obl_args) => match k {
                ty::Adt(impl_def, impl_args) => {
                    obl_def == impl_def && self.args_may_unify(obl_args, impl_args)
                }
                _ => false,
            },
            ty::Pat(obl_ty, _) => {
                // FIXME(pattern_types): take pattern into account
                matches!(k, ty::Pat(impl_ty, _) if self.types_may_unify(obl_ty, impl_ty))
            }
            ty::Slice(obl_ty) => {
                matches!(k, ty::Slice(impl_ty) if self.types_may_unify(obl_ty, impl_ty))
            }
            ty::Array(obl_ty, obl_len) => match k {
                ty::Array(impl_ty, impl_len) => {
                    self.types_may_unify(obl_ty, impl_ty)
                        && self.consts_may_unify(obl_len, impl_len)
                }
                _ => false,
            },
            ty::Tuple(obl) => match k {
                ty::Tuple(imp) => {
                    obl.len() == imp.len()
                        && iter::zip(obl.iter(), imp.iter())
                            .all(|(obl, imp)| self.types_may_unify(obl, imp))
                }
                _ => false,
            },
            ty::RawPtr(obl_ty, obl_mutbl) => match k {
                ty::RawPtr(imp_ty, imp_mutbl) => {
                    obl_mutbl == imp_mutbl && self.types_may_unify(obl_ty, imp_ty)
                }
                _ => false,
            },
            ty::Dynamic(obl_preds, ..) => {
                // Ideally we would walk the existential predicates here or at least
                // compare their length. But considering that the relevant `Relate` impl
                // actually sorts and deduplicates these, that doesn't work.
                matches!(k, ty::Dynamic(impl_preds, ..) if
                    obl_preds.principal_def_id() == impl_preds.principal_def_id()
                )
            }
            ty::FnPtr(obl_sig) => match k {
                ty::FnPtr(impl_sig) => {
                    let ty::FnSig { inputs_and_output, c_variadic, safety, abi } =
                        obl_sig.skip_binder();
                    let impl_sig = impl_sig.skip_binder();

                    abi == impl_sig.abi
                        && c_variadic == impl_sig.c_variadic
                        && safety == impl_sig.safety
                        && inputs_and_output.len() == impl_sig.inputs_and_output.len()
                        && iter::zip(inputs_and_output.iter(), impl_sig.inputs_and_output.iter())
                            .all(|(obl, imp)| self.types_may_unify(obl, imp))
                }
                _ => false,
            },

            // Impls cannot contain these types as these cannot be named directly.
            ty::FnDef(..) | ty::Closure(..) | ty::CoroutineClosure(..) | ty::Coroutine(..) => false,

            // Placeholder types don't unify with anything on their own
            ty::Placeholder(..) | ty::Bound(..) => false,

            // Depending on the value of `treat_obligation_params`, we either
            // treat generic parameters like placeholders or like inference variables.
            ty::Param(_) => match self.treat_obligation_params {
                TreatParams::ForLookup => false,
                TreatParams::AsCandidateKey => true,
            },

            ty::Infer(ty::IntVar(_)) => impl_ty.is_integral(),

            ty::Infer(ty::FloatVar(_)) => impl_ty.is_floating_point(),

            ty::Infer(_) => true,

            // As we're walking the whole type, it may encounter projections
            // inside of binders and what not, so we're just going to assume that
            // projections can unify with other stuff.
            //
            // Looking forward to lazy normalization this is the safer strategy anyways.
            ty::Alias(..) => true,

            ty::Error(_) => true,

            ty::CoroutineWitness(..) => {
                panic!("unexpected obligation type: {:?}", obligation_ty)
            }
        }
    }

    pub fn consts_may_unify(self, obligation_ct: I::Const, impl_ct: I::Const) -> bool {
        let impl_val = match impl_ct.kind() {
            ty::ConstKind::Expr(_)
            | ty::ConstKind::Param(_)
            | ty::ConstKind::Unevaluated(_)
            | ty::ConstKind::Error(_) => {
                return true;
            }
            ty::ConstKind::Value(_, impl_val) => impl_val,
            ty::ConstKind::Infer(_) | ty::ConstKind::Bound(..) | ty::ConstKind::Placeholder(_) => {
                panic!("unexpected impl arg: {:?}", impl_ct)
            }
        };

        match obligation_ct.kind() {
            ty::ConstKind::Param(_) => match self.treat_obligation_params {
                TreatParams::ForLookup => false,
                TreatParams::AsCandidateKey => true,
            },

            // Placeholder consts don't unify with anything on their own
            ty::ConstKind::Placeholder(_) => false,

            // As we don't necessarily eagerly evaluate constants,
            // they might unify with any value.
            ty::ConstKind::Expr(_) | ty::ConstKind::Unevaluated(_) | ty::ConstKind::Error(_) => {
                true
            }
            ty::ConstKind::Value(_, obl_val) => obl_val == impl_val,

            ty::ConstKind::Infer(_) => true,

            ty::ConstKind::Bound(..) => {
                panic!("unexpected obl const: {:?}", obligation_ct)
            }
        }
    }
}
