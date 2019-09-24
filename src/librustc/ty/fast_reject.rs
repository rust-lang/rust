use crate::hir::def_id::DefId;
use crate::ich::StableHashingContext;
use rustc_data_structures::stable_hasher::{StableHasher, StableHasherResult,
                                           HashStable};
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use syntax::ast;
use crate::ty::{self, Ty, TyCtxt};

use self::SimplifiedTypeGen::*;

pub type SimplifiedType = SimplifiedTypeGen<DefId>;

/// See `simplify_type`
///
/// Note that we keep this type generic over the type of identifier it uses
/// because we sometimes need to use SimplifiedTypeGen values as stable sorting
/// keys (in which case we use a DefPathHash as id-type) but in the general case
/// the non-stable but fast to construct DefId-version is the better choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, RustcEncodable, RustcDecodable)]
pub enum SimplifiedTypeGen<D>
    where D: Copy + Debug + Ord + Eq + Hash
{
    BoolSimplifiedType,
    CharSimplifiedType,
    IntSimplifiedType(ast::IntTy),
    UintSimplifiedType(ast::UintTy),
    FloatSimplifiedType(ast::FloatTy),
    AdtSimplifiedType(D),
    StrSimplifiedType,
    ArraySimplifiedType,
    PtrSimplifiedType,
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
    ForeignSimplifiedType(DefId),
}

/// Tries to simplify a type by dropping type parameters, deref'ing away any reference types, etc.
/// The idea is to get something simple that we can use to quickly decide if two types could unify
/// during method lookup.
///
/// If `can_simplify_params` is false, then we will fail to simplify type parameters entirely. This
/// is useful when those type parameters would be instantiated with fresh type variables, since
/// then we can't say much about whether two types would unify. Put another way,
/// `can_simplify_params` should be true if type parameters appear free in `ty` and `false` if they
/// are to be considered bound.
pub fn simplify_type(
    tcx: TyCtxt<'_>,
    ty: Ty<'_>,
    can_simplify_params: bool,
) -> Option<SimplifiedType> {
    match ty.sty {
        ty::Bool => Some(BoolSimplifiedType),
        ty::Char => Some(CharSimplifiedType),
        ty::Int(int_type) => Some(IntSimplifiedType(int_type)),
        ty::Uint(uint_type) => Some(UintSimplifiedType(uint_type)),
        ty::Float(float_type) => Some(FloatSimplifiedType(float_type)),
        ty::Adt(def, _) => Some(AdtSimplifiedType(def.did)),
        ty::Str => Some(StrSimplifiedType),
        ty::Array(..) | ty::Slice(_) => Some(ArraySimplifiedType),
        ty::RawPtr(_) => Some(PtrSimplifiedType),
        ty::Dynamic(ref trait_info, ..) => {
            match trait_info.principal_def_id() {
                Some(principal_def_id) if !tcx.trait_is_auto(principal_def_id) => {
                    Some(TraitSimplifiedType(principal_def_id))
                }
                _ => Some(MarkerTraitObjectSimplifiedType)
            }
        }
        ty::Ref(_, ty, _) => {
            // since we introduce auto-refs during method lookup, we
            // just treat &T and T as equivalent from the point of
            // view of possibly unifying
            simplify_type(tcx, ty, can_simplify_params)
        }
        ty::FnDef(def_id, _) |
        ty::Closure(def_id, _) => {
            Some(ClosureSimplifiedType(def_id))
        }
        ty::Generator(def_id, _, _) => {
            Some(GeneratorSimplifiedType(def_id))
        }
        ty::GeneratorWitness(ref tys) => {
            Some(GeneratorWitnessSimplifiedType(tys.skip_binder().len()))
        }
        ty::Never => Some(NeverSimplifiedType),
        ty::Tuple(ref tys) => {
            Some(TupleSimplifiedType(tys.len()))
        }
        ty::FnPtr(ref f) => {
            Some(FunctionSimplifiedType(f.skip_binder().inputs().len()))
        }
        ty::UnnormalizedProjection(..) => bug!("only used with chalk-engine"),
        ty::Projection(_) | ty::Param(_) => {
            if can_simplify_params {
                // In normalized types, projections don't unify with
                // anything. when lazy normalization happens, this
                // will change. It would still be nice to have a way
                // to deal with known-not-to-unify-with-anything
                // projections (e.g., the likes of <__S as Encoder>::Error).
                Some(ParameterSimplifiedType)
            } else {
                None
            }
        }
        ty::Opaque(def_id, _) => {
            Some(OpaqueSimplifiedType(def_id))
        }
        ty::Foreign(def_id) => {
            Some(ForeignSimplifiedType(def_id))
        }
        ty::Placeholder(..) | ty::Bound(..) | ty::Infer(_) | ty::Error => None,
    }
}

impl<D: Copy + Debug + Ord + Eq + Hash> SimplifiedTypeGen<D> {
    pub fn map_def<U, F>(self, map: F) -> SimplifiedTypeGen<U>
        where F: Fn(D) -> U,
              U: Copy + Debug + Ord + Eq + Hash,
    {
        match self {
            BoolSimplifiedType => BoolSimplifiedType,
            CharSimplifiedType => CharSimplifiedType,
            IntSimplifiedType(t) => IntSimplifiedType(t),
            UintSimplifiedType(t) => UintSimplifiedType(t),
            FloatSimplifiedType(t) => FloatSimplifiedType(t),
            AdtSimplifiedType(d) => AdtSimplifiedType(map(d)),
            StrSimplifiedType => StrSimplifiedType,
            ArraySimplifiedType => ArraySimplifiedType,
            PtrSimplifiedType => PtrSimplifiedType,
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
            ForeignSimplifiedType(d) => ForeignSimplifiedType(d),
        }
    }
}

impl<'a, D> HashStable<StableHashingContext<'a>> for SimplifiedTypeGen<D>
where
    D: Copy + Debug + Ord + Eq + Hash + HashStable<StableHashingContext<'a>>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            BoolSimplifiedType |
            CharSimplifiedType |
            StrSimplifiedType |
            ArraySimplifiedType |
            PtrSimplifiedType |
            NeverSimplifiedType |
            ParameterSimplifiedType |
            MarkerTraitObjectSimplifiedType => {
                // nothing to do
            }
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
