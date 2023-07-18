use crate::mir::Mutability;
use crate::ty::GenericArgKind;
use crate::ty::{self, GenericArgsRef, Ty, TyCtxt, TypeVisitableExt};
use rustc_hir::def_id::DefId;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;

/// See `simplify_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, TyEncodable, TyDecodable, HashStable)]
pub enum SimplifiedType {
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
    Generator(DefId),
    GeneratorWitness(usize),
    GeneratorWitnessMIR(DefId),
    Function(usize),
    Placeholder,
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
    /// Treat parameters as placeholders in the given environment. This is the
    /// correct mode for *lookup*, as during candidate selection.
    ///
    /// N.B. during deep rejection, this acts identically to `ForLookup`.
    ///
    /// FIXME(-Ztrait-solver=next): Remove this variant and cleanup
    /// the code.
    NextSolverLookup,
}

/// During fast-rejection, we have the choice of treating projection types
/// as either simplifiable or not, depending on whether we expect the projection
/// to be normalized/rigid.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum TreatProjections {
    /// In the old solver we don't try to normalize projections
    /// when looking up impls and only access them by using the
    /// current self type. This means that if the self type is
    /// a projection which could later be normalized, we must not
    /// treat it as rigid.
    ForLookup,
    /// We can treat projections in the self type as opaque as
    /// we separately look up impls for the normalized self type.
    NextSolverLookup,
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
        ty::Bool => Some(SimplifiedType::Bool),
        ty::Char => Some(SimplifiedType::Char),
        ty::Int(int_type) => Some(SimplifiedType::Int(int_type)),
        ty::Uint(uint_type) => Some(SimplifiedType::Uint(uint_type)),
        ty::Float(float_type) => Some(SimplifiedType::Float(float_type)),
        ty::Adt(def, _) => Some(SimplifiedType::Adt(def.did())),
        ty::Str => Some(SimplifiedType::Str),
        ty::Array(..) => Some(SimplifiedType::Array),
        ty::Slice(..) => Some(SimplifiedType::Slice),
        ty::RawPtr(ptr) => Some(SimplifiedType::Ptr(ptr.mutbl)),
        ty::Dynamic(trait_info, ..) => match trait_info.principal_def_id() {
            Some(principal_def_id) if !tcx.trait_is_auto(principal_def_id) => {
                Some(SimplifiedType::Trait(principal_def_id))
            }
            _ => Some(SimplifiedType::MarkerTraitObject),
        },
        ty::Ref(_, _, mutbl) => Some(SimplifiedType::Ref(mutbl)),
        ty::FnDef(def_id, _) | ty::Closure(def_id, _) => Some(SimplifiedType::Closure(def_id)),
        ty::Generator(def_id, _, _) => Some(SimplifiedType::Generator(def_id)),
        ty::GeneratorWitness(tys) => {
            Some(SimplifiedType::GeneratorWitness(tys.skip_binder().len()))
        }
        ty::GeneratorWitnessMIR(def_id, _) => Some(SimplifiedType::GeneratorWitnessMIR(def_id)),
        ty::Never => Some(SimplifiedType::Never),
        ty::Tuple(tys) => Some(SimplifiedType::Tuple(tys.len())),
        ty::FnPtr(f) => Some(SimplifiedType::Function(f.skip_binder().inputs().len())),
        ty::Placeholder(..) => Some(SimplifiedType::Placeholder),
        ty::Param(_) => match treat_params {
            TreatParams::ForLookup | TreatParams::NextSolverLookup => {
                Some(SimplifiedType::Placeholder)
            }
            TreatParams::AsCandidateKey => None,
        },
        ty::Alias(..) => match treat_params {
            // When treating `ty::Param` as a placeholder, projections also
            // don't unify with anything else as long as they are fully normalized.
            //
            // We will have to be careful with lazy normalization here.
            // FIXME(lazy_normalization): This is probably not right...
            TreatParams::ForLookup if !ty.has_non_region_infer() => {
                Some(SimplifiedType::Placeholder)
            }
            TreatParams::NextSolverLookup => Some(SimplifiedType::Placeholder),
            TreatParams::ForLookup | TreatParams::AsCandidateKey => None,
        },
        ty::Foreign(def_id) => Some(SimplifiedType::Foreign(def_id)),
        ty::Bound(..) | ty::Infer(_) | ty::Error(_) => None,
    }
}

impl SimplifiedType {
    pub fn def(self) -> Option<DefId> {
        match self {
            SimplifiedType::Adt(d)
            | SimplifiedType::Foreign(d)
            | SimplifiedType::Trait(d)
            | SimplifiedType::Closure(d)
            | SimplifiedType::Generator(d)
            | SimplifiedType::GeneratorWitnessMIR(d) => Some(d),
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
    pub fn args_refs_may_unify<'tcx>(
        self,
        obligation_args: GenericArgsRef<'tcx>,
        impl_args: GenericArgsRef<'tcx>,
    ) -> bool {
        iter::zip(obligation_args, impl_args).all(|(obl, imp)| {
            match (obl.unpack(), imp.unpack()) {
                // We don't fast reject based on regions for now.
                (GenericArgKind::Lifetime(_), GenericArgKind::Lifetime(_)) => true,
                (GenericArgKind::Type(obl), GenericArgKind::Type(imp)) => {
                    self.types_may_unify(obl, imp)
                }
                (GenericArgKind::Const(obl), GenericArgKind::Const(imp)) => {
                    self.consts_may_unify(obl, imp)
                }
                _ => bug!("kind mismatch: {obl} {imp}"),
            }
        })
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
            ty::Adt(obl_def, obl_args) => match k {
                &ty::Adt(impl_def, impl_args) => {
                    obl_def == impl_def && self.args_refs_may_unify(obl_args, impl_args)
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

            // Placeholder types don't unify with anything on their own
            ty::Placeholder(..) | ty::Bound(..) => false,

            // Depending on the value of `treat_obligation_params`, we either
            // treat generic parameters like placeholders or like inference variables.
            ty::Param(_) => match self.treat_obligation_params {
                TreatParams::ForLookup | TreatParams::NextSolverLookup => false,
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
                TreatParams::ForLookup | TreatParams::NextSolverLookup => false,
                TreatParams::AsCandidateKey => true,
            },

            // Placeholder consts don't unify with anything on their own
            ty::ConstKind::Placeholder(_) => false,

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

            ty::ConstKind::Bound(..) => {
                bug!("unexpected obl const: {:?}", obligation_ct)
            }
        }
    }
}
