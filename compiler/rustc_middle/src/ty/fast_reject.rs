use crate::mir::Mutability;
use crate::ty::{self, Ty, TyCtxt};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::DefId;
use rustc_query_system::ich::StableHashingContext;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;

use self::SimplifiedTypeGen::*;

pub type SimplifiedType = SimplifiedTypeGen<DefId>;

/// See `simplify_type`
///
/// Note that we keep this type generic over the type of identifier it uses
/// because we sometimes need to use SimplifiedTypeGen values as stable sorting
/// keys (in which case we use a DefPathHash as id-type) but in the general case
/// the non-stable but fast to construct DefId-version is the better choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
pub enum SimplifiedTypeGen<D>
where
    D: Copy + Debug + Eq,
{
    BoolSimplifiedType,
    CharSimplifiedType,
    IntSimplifiedType(ty::IntTy),
    UintSimplifiedType(ty::UintTy),
    FloatSimplifiedType(ty::FloatTy),
    AdtSimplifiedType(D),
    ForeignSimplifiedType(D),
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
    TraitSimplifiedType(D),
    ClosureSimplifiedType(D),
    GeneratorSimplifiedType(D),
    GeneratorWitnessSimplifiedType(usize),
    OpaqueSimplifiedType(D),
    FunctionSimplifiedType(usize),
    ParameterSimplifiedType,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum TreatParams {
    /// Treat parameters as bound types in the given environment.
    ///
    /// For this to be correct the input has to be fully normalized
    /// in its param env as it may otherwise cause us to ignore
    /// potentially applying impls.
    AsBoundTypes,
    AsPlaceholders,
}

/// Tries to simplify a type by only returning the outermost injective¹ layer, if one exists.
///
/// The idea is to get something simple that we can use to quickly decide if two types could unify,
/// for example during method lookup.
///
/// A special case here are parameters and projections, which are only injective
/// if they are treated as bound types.
///
/// For example when storing impls based on their simplified self type, we treat
/// generic parameters as placeholders. We must not simplify them here,
/// as they can unify with any other type.
///
/// With projections we have to be even more careful, as even when treating them as bound types
/// this is still only correct if they are fully normalized.
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
        ty::Never => Some(NeverSimplifiedType),
        ty::Tuple(tys) => Some(TupleSimplifiedType(tys.len())),
        ty::FnPtr(f) => Some(FunctionSimplifiedType(f.skip_binder().inputs().len())),
        ty::Param(_) | ty::Projection(_) => match treat_params {
            // When treated as bound types, projections don't unify with
            // anything as long as they are fully normalized.
            //
            // We will have to be careful with lazy normalization here.
            TreatParams::AsBoundTypes => {
                debug!("treating `{}` as a bound type", ty);
                Some(ParameterSimplifiedType)
            }
            TreatParams::AsPlaceholders => None,
        },
        ty::Opaque(def_id, _) => Some(OpaqueSimplifiedType(def_id)),
        ty::Foreign(def_id) => Some(ForeignSimplifiedType(def_id)),
        ty::Placeholder(..) | ty::Bound(..) | ty::Infer(_) | ty::Error(_) => None,
    }
}

impl<D: Copy + Debug + Eq> SimplifiedTypeGen<D> {
    pub fn def(self) -> Option<D> {
        match self {
            AdtSimplifiedType(d)
            | ForeignSimplifiedType(d)
            | TraitSimplifiedType(d)
            | ClosureSimplifiedType(d)
            | GeneratorSimplifiedType(d)
            | OpaqueSimplifiedType(d) => Some(d),
            _ => None,
        }
    }

    pub fn map_def<U, F>(self, map: F) -> SimplifiedTypeGen<U>
    where
        F: Fn(D) -> U,
        U: Copy + Debug + Eq,
    {
        match self {
            BoolSimplifiedType => BoolSimplifiedType,
            CharSimplifiedType => CharSimplifiedType,
            IntSimplifiedType(t) => IntSimplifiedType(t),
            UintSimplifiedType(t) => UintSimplifiedType(t),
            FloatSimplifiedType(t) => FloatSimplifiedType(t),
            AdtSimplifiedType(d) => AdtSimplifiedType(map(d)),
            ForeignSimplifiedType(d) => ForeignSimplifiedType(map(d)),
            StrSimplifiedType => StrSimplifiedType,
            ArraySimplifiedType => ArraySimplifiedType,
            SliceSimplifiedType => SliceSimplifiedType,
            RefSimplifiedType(m) => RefSimplifiedType(m),
            PtrSimplifiedType(m) => PtrSimplifiedType(m),
            NeverSimplifiedType => NeverSimplifiedType,
            MarkerTraitObjectSimplifiedType => MarkerTraitObjectSimplifiedType,
            TupleSimplifiedType(n) => TupleSimplifiedType(n),
            TraitSimplifiedType(d) => TraitSimplifiedType(map(d)),
            ClosureSimplifiedType(d) => ClosureSimplifiedType(map(d)),
            GeneratorSimplifiedType(d) => GeneratorSimplifiedType(map(d)),
            GeneratorWitnessSimplifiedType(n) => GeneratorWitnessSimplifiedType(n),
            OpaqueSimplifiedType(d) => OpaqueSimplifiedType(map(d)),
            FunctionSimplifiedType(n) => FunctionSimplifiedType(n),
            ParameterSimplifiedType => ParameterSimplifiedType,
        }
    }
}

impl<'a, D> HashStable<StableHashingContext<'a>> for SimplifiedTypeGen<D>
where
    D: Copy + Debug + Eq + HashStable<StableHashingContext<'a>>,
{
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            BoolSimplifiedType
            | CharSimplifiedType
            | StrSimplifiedType
            | ArraySimplifiedType
            | SliceSimplifiedType
            | NeverSimplifiedType
            | ParameterSimplifiedType
            | MarkerTraitObjectSimplifiedType => {
                // nothing to do
            }
            RefSimplifiedType(m) | PtrSimplifiedType(m) => m.hash_stable(hcx, hasher),
            IntSimplifiedType(t) => t.hash_stable(hcx, hasher),
            UintSimplifiedType(t) => t.hash_stable(hcx, hasher),
            FloatSimplifiedType(t) => t.hash_stable(hcx, hasher),
            AdtSimplifiedType(d) => d.hash_stable(hcx, hasher),
            TupleSimplifiedType(n) => n.hash_stable(hcx, hasher),
            TraitSimplifiedType(d) => d.hash_stable(hcx, hasher),
            ClosureSimplifiedType(d) => d.hash_stable(hcx, hasher),
            GeneratorSimplifiedType(d) => d.hash_stable(hcx, hasher),
            GeneratorWitnessSimplifiedType(n) => n.hash_stable(hcx, hasher),
            OpaqueSimplifiedType(d) => d.hash_stable(hcx, hasher),
            FunctionSimplifiedType(n) => n.hash_stable(hcx, hasher),
            ForeignSimplifiedType(d) => d.hash_stable(hcx, hasher),
        }
    }
}
