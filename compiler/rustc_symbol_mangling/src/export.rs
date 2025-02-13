use std::assert_matches::debug_assert_matches;

use rustc_abi::IntegerType;
use rustc_data_structures::stable_hasher::{Hash128, StableHasher};
use rustc_hir::def::DefKind;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_span::symbol::{Symbol, sym};

trait AbiHashStable<'tcx> {
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher);
}
macro_rules! default_hash_impl {
    ($t:ty) => {
        impl<'tcx> AbiHashStable<'tcx> for $t {
            #[inline]
            fn hash(&self, _tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
                ::std::hash::Hash::hash(self, hasher);
            }
        }
    };
}

default_hash_impl!(i8);
default_hash_impl!(i16);
default_hash_impl!(i32);
default_hash_impl!(i64);
default_hash_impl!(i128);
default_hash_impl!(isize);

default_hash_impl!(u8);
default_hash_impl!(u16);
default_hash_impl!(u32);
default_hash_impl!(u64);
default_hash_impl!(u128);
default_hash_impl!(usize);

impl<'tcx> AbiHashStable<'tcx> for bool {
    #[inline]
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        (if *self { 1u8 } else { 0u8 }).hash(tcx, hasher);
    }
}

impl<'tcx> AbiHashStable<'tcx> for str {
    #[inline]
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        self.as_bytes().hash(tcx, hasher);
    }
}

impl<'tcx> AbiHashStable<'tcx> for String {
    #[inline]
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        self[..].hash(tcx, hasher);
    }
}

impl<'tcx> AbiHashStable<'tcx> for Symbol {
    #[inline]
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        self.as_str().hash(tcx, hasher);
    }
}

impl<'tcx, T: AbiHashStable<'tcx>> AbiHashStable<'tcx> for [T] {
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        self.len().hash(tcx, hasher);
        for item in self {
            item.hash(tcx, hasher);
        }
    }
}

impl<'tcx> AbiHashStable<'tcx> for Ty<'tcx> {
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        match self.kind() {
            ty::Bool => sym::bool.hash(tcx, hasher),
            ty::Char => sym::char.hash(tcx, hasher),
            ty::Int(int_ty) => int_ty.name_str().hash(tcx, hasher),
            ty::Uint(uint_ty) => uint_ty.name_str().hash(tcx, hasher),
            ty::Float(float_ty) => float_ty.name_str().hash(tcx, hasher),

            ty::Adt(adt_def, args) => {
                adt_def.is_struct().hash(tcx, hasher);
                adt_def.is_enum().hash(tcx, hasher);
                adt_def.is_union().hash(tcx, hasher);

                if let Some(align) = adt_def.repr().align {
                    align.bits().hash(tcx, hasher);
                }

                if let Some(integer) = adt_def.repr().int {
                    match integer {
                        IntegerType::Pointer(sign) => sign.hash(tcx, hasher),
                        IntegerType::Fixed(integer, sign) => {
                            integer.int_ty_str().hash(tcx, hasher);
                            sign.hash(tcx, hasher);
                        }
                    }
                }

                if let Some(pack) = adt_def.repr().pack {
                    pack.bits().hash(tcx, hasher);
                }

                adt_def.repr().c().hash(tcx, hasher);

                for variant in adt_def.variants() {
                    variant.name.hash(tcx, hasher);
                    for field in &variant.fields {
                        field.name.hash(tcx, hasher);
                        let field_ty = tcx.type_of(field.did).instantiate_identity();
                        field_ty.hash(tcx, hasher);
                    }
                }
                args.hash(tcx, hasher);
            }

            // FIXME: Not yet supported.
            ty::Foreign(_)
            | ty::Ref(_, _, _)
            | ty::Str
            | ty::Array(_, _)
            | ty::Pat(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_, _)
            | ty::Dynamic(_, _, _)
            | ty::Closure(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(_, _)
            | ty::Never
            | ty::Tuple(_)
            | ty::Alias(_, _)
            | ty::Param(_)
            | ty::Bound(_, _)
            | ty::Placeholder(_)
            | ty::Infer(_)
            | ty::UnsafeBinder(_) => unreachable!(),

            ty::Error(_) => {}
        }
    }
}

impl<'tcx> AbiHashStable<'tcx> for ty::FnSig<'tcx> {
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        for ty in self.inputs_and_output {
            ty.hash(tcx, hasher);
        }
        self.safety.is_safe().hash(tcx, hasher);
    }
}

impl<'tcx> AbiHashStable<'tcx> for ty::GenericArg<'tcx> {
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        self.unpack().hash(tcx, hasher);
    }
}

impl<'tcx> AbiHashStable<'tcx> for ty::GenericArgKind<'tcx> {
    fn hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        match self {
            ty::GenericArgKind::Lifetime(r) => r.hash(tcx, hasher),
            ty::GenericArgKind::Type(t) => t.hash(tcx, hasher),
            ty::GenericArgKind::Const(c) => c.hash(tcx, hasher),
        }
    }
}

impl<'tcx> AbiHashStable<'tcx> for ty::Const<'tcx> {
    fn hash(&self, _tcx: TyCtxt<'tcx>, _hasher: &mut StableHasher) {
        unimplemented!()
    }
}

impl<'tcx> AbiHashStable<'tcx> for ty::Region<'tcx> {
    fn hash(&self, _tcx: TyCtxt<'tcx>, _hasher: &mut StableHasher) {
        unimplemented!()
    }
}

pub(crate) fn compute_hash_of_export_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> String {
    let def_id = instance.def_id();
    debug_assert_matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn);

    let args = instance.args;
    let sig_ty = tcx.fn_sig(def_id).instantiate(tcx, args);
    let sig_ty = tcx.instantiate_bound_regions_with_erased(sig_ty);

    let hash = {
        let mut hasher = StableHasher::new();
        sig_ty.hash(tcx, &mut hasher);
        hasher.finish::<Hash128>()
    };

    hash.as_u128().to_string()
}
