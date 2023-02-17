use crate::mir::Mutability;
use crate::ty::subst::GenericArgKind;
use crate::ty::{self, Ty, TyCtxt, TypeVisitable};
use rustc_hir::def_id::DefId;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;

use self::SimplifiedType::*;

/// See `simplify_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, TyEncodable, TyDecodable, HashStable)]
pub enum SimplifiedType {
    BoolSimplifiedType,
    CharSimplifiedType,
    IntSimplifiedType(ty::IntTy),
    UintSimplifiedType(ty::UintTy),
    FloatSimplifiedType(ty::FloatTy),
    AdtSimplifiedType(DefId),
    ForeignSimplifiedType(DefId),
    StrSimplifiedType,
    ArraySimplifiedType,
    SliceSimplifiedType,
    RefSimplifiedType(Mutability),
    PtrSimplifiedType(Mutability),
    NeverSimplifiedType,
    TupleSimplifiedType(usize),
    /// A trait object, all of whose components are markers
    /// (e.g., `dyn Send + Sync`).
    MarkerTraitObjectSimplifiedType,
    TraitSimplifiedType(DefId),
    ClosureSimplifiedType(DefId),
    GeneratorSimplifiedType(DefId),
    GeneratorWitnessSimplifiedType(usize),
    GeneratorWitnessMIRSimplifiedType(DefId),
    FunctionSimplifiedType(usize),
    PlaceholderSimplifiedType,
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
    /// Treat parameters as placeholders in the given environment.
    ///
    /// Note that this also causes us to treat projections as if they were
    /// placeholders. This is only correct if the given projection cannot
    /// be normalized in the current context. Even if normalization fails,
    /// it may still succeed later if the projection contains any inference
    /// variables.
    AsPlaceholder,
    AsInfer,
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
pub fn simplify_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    treat_params: TreatParams,
) -> Option<SimplifiedType> {
    match *ty.kind() {
        ty::Bool => Some(BoolSimplifiedType),
        ty::Char => Some(CharSimplifiedType),
        ty::Int(int_type) => Some(IntSimplifiedType(int_type)),
        ty::Uint(uint_type) => Some(UintSimplifiedType(uint_type)),
        ty::Float(float_type) => Some(FloatSimplifiedType(float_type)),
        ty::Adt(def, _) => Some(AdtSimplifiedType(def.did())),
        ty::Str => Some(StrSimplifiedType),
        ty::Array(..) => Some(ArraySimplifiedType),
        ty::Slice(..) => Some(SliceSimplifiedType),
        ty::RawPtr(ptr) => Some(PtrSimplifiedType(ptr.mutbl)),
        ty::Dynamic(trait_info, ..) => match trait_info.principal_def_id() {
            Some(principal_def_id) if !tcx.trait_is_auto(principal_def_id) => {
                Some(TraitSimplifiedType(principal_def_id))
            }
            _ => Some(MarkerTraitObjectSimplifiedType),
        },
        ty::Ref(_, _, mutbl) => Some(RefSimplifiedType(mutbl)),
        ty::FnDef(def_id, _) | ty::Closure(def_id, _) => Some(ClosureSimplifiedType(def_id)),
        ty::Generator(def_id, _, _) => Some(GeneratorSimplifiedType(def_id)),
        ty::GeneratorWitness(tys) => Some(GeneratorWitnessSimplifiedType(tys.skip_binder().len())),
        ty::GeneratorWitnessMIR(def_id, _) => Some(GeneratorWitnessMIRSimplifiedType(def_id)),
        ty::Never => Some(NeverSimplifiedType),
        ty::Tuple(tys) => Some(TupleSimplifiedType(tys.len())),
        ty::FnPtr(f) => Some(FunctionSimplifiedType(f.skip_binder().inputs().len())),
        ty::Placeholder(..) => Some(PlaceholderSimplifiedType),
        ty::Param(_) => match treat_params {
            TreatParams::AsPlaceholder => Some(PlaceholderSimplifiedType),
            TreatParams::AsInfer => None,
        },
        ty::Alias(..) => match treat_params {
            // When treating `ty::Param` as a placeholder, projections also
            // don't unify with anything else as long as they are fully normalized.
            //
            // We will have to be careful with lazy normalization here.
            TreatParams::AsPlaceholder if !ty.has_non_region_infer() => {
                debug!("treating `{}` as a placeholder", ty);
                Some(PlaceholderSimplifiedType)
            }
            TreatParams::AsPlaceholder | TreatParams::AsInfer => None,
        },
        ty::Foreign(def_id) => Some(ForeignSimplifiedType(def_id)),
        ty::Bound(..) | ty::Infer(_) | ty::Error(_) => None,
    }
}

impl SimplifiedType {
    pub fn def(self) -> Option<DefId> {
        match self {
            AdtSimplifiedType(d)
            | ForeignSimplifiedType(d)
            | TraitSimplifiedType(d)
            | ClosureSimplifiedType(d)
            | GeneratorSimplifiedType(d)
            | GeneratorWitnessMIRSimplifiedType(d) => Some(d),
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
pub struct DeepRejectCtxt {
    pub treat_obligation_params: TreatParams,
}

impl DeepRejectCtxt {
    pub fn generic_args_may_unify<'tcx>(
        self,
        obligation_arg: ty::GenericArg<'tcx>,
        impl_arg: ty::GenericArg<'tcx>,
    ) -> bool {
        match (obligation_arg.unpack(), impl_arg.unpack()) {
            // We don't fast reject based on regions for now.
            (GenericArgKind::Lifetime(_), GenericArgKind::Lifetime(_)) => true,
            (GenericArgKind::Type(obl), GenericArgKind::Type(imp)) => {
                self.types_may_unify(obl, imp)
            }
            (GenericArgKind::Const(obl), GenericArgKind::Const(imp)) => {
                self.consts_may_unify(obl, imp)
            }
            _ => bug!("kind mismatch: {obligation_arg} {impl_arg}"),
        }
    }

    pub fn types_may_unify<'tcx>(self, obligation_ty: Ty<'tcx>, impl_ty: Ty<'tcx>) -> bool {
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
            | ty::Ref(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::FnPtr(..)
            | ty::Foreign(..) => {}
            ty::FnDef(..)
            | ty::Closure(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::GeneratorWitnessMIR(..)
            | ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Infer(_) => bug!("unexpected impl_ty: {impl_ty}"),
        }

        let k = impl_ty.kind();
        match *obligation_ty.kind() {
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
                &ty::Ref(_, impl_ty, impl_mutbl) => {
                    obl_mutbl == impl_mutbl && self.types_may_unify(obl_ty, impl_ty)
                }
                _ => false,
            },
            ty::Adt(obl_def, obl_substs) => match k {
                &ty::Adt(impl_def, impl_substs) => {
                    obl_def == impl_def
                        && iter::zip(obl_substs, impl_substs)
                            .all(|(obl, imp)| self.generic_args_may_unify(obl, imp))
                }
                _ => false,
            },
            ty::Slice(obl_ty) => {
                matches!(k, &ty::Slice(impl_ty) if self.types_may_unify(obl_ty, impl_ty))
            }
            ty::Array(obl_ty, obl_len) => match k {
                &ty::Array(impl_ty, impl_len) => {
                    self.types_may_unify(obl_ty, impl_ty)
                        && self.consts_may_unify(obl_len, impl_len)
                }
                _ => false,
            },
            ty::Tuple(obl) => match k {
                &ty::Tuple(imp) => {
                    obl.len() == imp.len()
                        && iter::zip(obl, imp).all(|(obl, imp)| self.types_may_unify(obl, imp))
                }
                _ => false,
            },
            ty::RawPtr(obl) => match k {
                ty::RawPtr(imp) => obl.mutbl == imp.mutbl && self.types_may_unify(obl.ty, imp.ty),
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
                    let ty::FnSig { inputs_and_output, c_variadic, unsafety, abi } =
                        obl_sig.skip_binder();
                    let impl_sig = impl_sig.skip_binder();

                    abi == impl_sig.abi
                        && c_variadic == impl_sig.c_variadic
                        && unsafety == impl_sig.unsafety
                        && inputs_and_output.len() == impl_sig.inputs_and_output.len()
                        && iter::zip(inputs_and_output, impl_sig.inputs_and_output)
                            .all(|(obl, imp)| self.types_may_unify(obl, imp))
                }
                _ => false,
            },

            // Impls cannot contain these types as these cannot be named directly.
            ty::FnDef(..) | ty::Closure(..) | ty::Generator(..) => false,

            ty::Placeholder(..) | ty::Bound(..) => false,

            // Depending on the value of `treat_obligation_params`, we either
            // treat generic parameters like placeholders or like inference variables.
            ty::Param(_) => match self.treat_obligation_params {
                TreatParams::AsPlaceholder => false,
                TreatParams::AsInfer => true,
            },

            ty::Infer(_) => true,

            // As we're walking the whole type, it may encounter projections
            // inside of binders and what not, so we're just going to assume that
            // projections can unify with other stuff.
            //
            // Looking forward to lazy normalization this is the safer strategy anyways.
            ty::Alias(..) => true,

            ty::Error(_) => true,

            ty::GeneratorWitness(..) | ty::GeneratorWitnessMIR(..) => {
                bug!("unexpected obligation type: {:?}", obligation_ty)
            }
        }
    }

    pub fn consts_may_unify(self, obligation_ct: ty::Const<'_>, impl_ct: ty::Const<'_>) -> bool {
        match impl_ct.kind() {
            ty::ConstKind::Expr(_)
            | ty::ConstKind::Param(_)
            | ty::ConstKind::Unevaluated(_)
            | ty::ConstKind::Error(_) => {
                return true;
            }
            ty::ConstKind::Value(_) => {}
            ty::ConstKind::Infer(_) | ty::ConstKind::Bound(..) | ty::ConstKind::Placeholder(_) => {
                bug!("unexpected impl arg: {:?}", impl_ct)
            }
        }

        let k = impl_ct.kind();
        match obligation_ct.kind() {
            ty::ConstKind::Param(_) => match self.treat_obligation_params {
                TreatParams::AsPlaceholder => false,
                TreatParams::AsInfer => true,
            },

            // As we don't necessarily eagerly evaluate constants,
            // they might unify with any value.
            ty::ConstKind::Expr(_) | ty::ConstKind::Unevaluated(_) | ty::ConstKind::Error(_) => {
                true
            }
            ty::ConstKind::Value(obl) => match k {
                ty::ConstKind::Value(imp) => obl == imp,
                _ => true,
            },

            ty::ConstKind::Infer(_) => true,

            ty::ConstKind::Bound(..) | ty::ConstKind::Placeholder(_) => {
                bug!("unexpected obl const: {:?}", obligation_ct)
            }
        }
    }
}
