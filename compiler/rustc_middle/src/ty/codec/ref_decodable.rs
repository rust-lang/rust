use std::marker::PointeeSized;

use rustc_abi::FieldIdx;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::Const;
use rustc_serialize::Decodable;

use crate::ty::codec::TyDecoder;
use crate::ty::{self, Ty};
use crate::{mir, traits};

/// Trait for decoding to a reference.
///
/// This is a separate trait from `Decodable` so that we can implement it for
/// upstream types, such as `FxHashSet`.
///
/// The `TyDecodable` derive macro will use this trait for fields that are
/// references (and don't use a type alias to hide that).
///
/// `Decodable` can still be implemented in cases where `Decodable` is required
/// by a trait bound.
///
/// Implementations of this trait will typically allocate into an arena or interner,
/// e.g. see `impl_ref_decodable_into_arena!`.
pub trait RefDecodable<'tcx, D: TyDecoder<'tcx>>: PointeeSized {
    fn decode(d: &mut D) -> &'tcx Self;
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::List<Ty<'tcx>> {
    fn decode(decoder: &mut D) -> &'tcx Self {
        let len = decoder.read_usize();
        decoder
            .interner()
            .mk_type_list_from_iter((0..len).map::<Ty<'tcx>, _>(|_| Decodable::decode(decoder)))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D>
    for ty::List<ty::PolyExistentialPredicate<'tcx>>
{
    fn decode(decoder: &mut D) -> &'tcx Self {
        let len = decoder.read_usize();
        decoder.interner().mk_poly_existential_predicates_from_iter(
            (0..len).map::<ty::Binder<'tcx, _>, _>(|_| Decodable::decode(decoder)),
        )
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::List<ty::BoundVariableKind<'tcx>> {
    fn decode(decoder: &mut D) -> &'tcx Self {
        let len = decoder.read_usize();
        decoder.interner().mk_bound_variable_kinds_from_iter(
            (0..len).map::<ty::BoundVariableKind<'tcx>, _>(|_| Decodable::decode(decoder)),
        )
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::List<ty::Pattern<'tcx>> {
    fn decode(decoder: &mut D) -> &'tcx Self {
        let len = decoder.read_usize();
        decoder.interner().mk_patterns_from_iter(
            (0..len).map::<ty::Pattern<'tcx>, _>(|_| Decodable::decode(decoder)),
        )
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::List<ty::Const<'tcx>> {
    fn decode(decoder: &mut D) -> &'tcx Self {
        let len = decoder.read_usize();
        decoder.interner().mk_const_list_from_iter(
            (0..len).map::<ty::Const<'tcx>, _>(|_| Decodable::decode(decoder)),
        )
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D>
    for ty::ListWithCachedTypeInfo<ty::Clause<'tcx>>
{
    fn decode(decoder: &mut D) -> &'tcx Self {
        let len = decoder.read_usize();
        decoder.interner().mk_clauses_from_iter(
            (0..len).map::<ty::Clause<'tcx>, _>(|_| Decodable::decode(decoder)),
        )
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::List<FieldIdx> {
    fn decode(decoder: &mut D) -> &'tcx Self {
        let len = decoder.read_usize();
        decoder
            .interner()
            .mk_fields_from_iter((0..len).map::<FieldIdx, _>(|_| Decodable::decode(decoder)))
    }
}

impl<'tcx, D: TyDecoder<'tcx>> RefDecodable<'tcx, D> for ty::List<LocalDefId> {
    fn decode(decoder: &mut D) -> &'tcx Self {
        let len = decoder.read_usize();
        decoder.interner().mk_local_def_ids_from_iter(
            (0..len).map::<LocalDefId, _>(|_| Decodable::decode(decoder)),
        )
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for &'tcx ty::List<LocalDefId> {
    fn decode(d: &mut D) -> Self {
        RefDecodable::decode(d)
    }
}

macro_rules! impl_decodable_via_ref {
    ($($t:ty,)+) => {
        $(impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for $t {
            fn decode(decoder: &mut D) -> Self {
                RefDecodable::decode(decoder)
            }
        })*
    }
}

impl_decodable_via_ref! {
    &'tcx ty::TypeckResults<'tcx>,
    &'tcx ty::List<Ty<'tcx>>,
    &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    &'tcx traits::ImplSource<'tcx, ()>,
    &'tcx mir::Body<'tcx>,
    &'tcx ty::List<ty::BoundVariableKind<'tcx>>,
    &'tcx ty::List<ty::Pattern<'tcx>>,
    &'tcx ty::ListWithCachedTypeInfo<ty::Clause<'tcx>>,
    &'tcx ty::List<Const<'tcx>>,
}
