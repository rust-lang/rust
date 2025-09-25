use std::assert_matches::debug_assert_matches;

use rustc_abi::IntegerType;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_hashes::Hash128;
use rustc_hir::def::DefKind;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_span::symbol::{Symbol, sym};

trait AbiHashStable<'tcx> {
    fn abi_hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher);
}
macro_rules! default_hash_impl {
    ($($t:ty,)+) => {
        $(impl<'tcx> AbiHashStable<'tcx> for $t {
            #[inline]
            fn abi_hash(&self, _tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
                ::std::hash::Hash::hash(self, hasher);
            }
        })*
    };
}

default_hash_impl! { u8, u64, usize, }

impl<'tcx> AbiHashStable<'tcx> for bool {
    #[inline]
    fn abi_hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        (if *self { 1u8 } else { 0u8 }).abi_hash(tcx, hasher);
    }
}

impl<'tcx> AbiHashStable<'tcx> for str {
    #[inline]
    fn abi_hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        self.as_bytes().abi_hash(tcx, hasher);
    }
}

impl<'tcx> AbiHashStable<'tcx> for Symbol {
    #[inline]
    fn abi_hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        self.as_str().abi_hash(tcx, hasher);
    }
}

impl<'tcx, T: AbiHashStable<'tcx>> AbiHashStable<'tcx> for [T] {
    fn abi_hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        self.len().abi_hash(tcx, hasher);
        for item in self {
            item.abi_hash(tcx, hasher);
        }
    }
}

impl<'tcx> AbiHashStable<'tcx> for Ty<'tcx> {
    fn abi_hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        match self.kind() {
            ty::Bool => sym::bool.abi_hash(tcx, hasher),
            ty::Char => sym::char.abi_hash(tcx, hasher),
            ty::Int(int_ty) => int_ty.name_str().abi_hash(tcx, hasher),
            ty::Uint(uint_ty) => uint_ty.name_str().abi_hash(tcx, hasher),
            ty::Float(float_ty) => float_ty.name_str().abi_hash(tcx, hasher),

            ty::Adt(adt_def, args) => {
                adt_def.is_struct().abi_hash(tcx, hasher);
                adt_def.is_enum().abi_hash(tcx, hasher);
                adt_def.is_union().abi_hash(tcx, hasher);

                if let Some(align) = adt_def.repr().align {
                    align.bits().abi_hash(tcx, hasher);
                }

                if let Some(integer) = adt_def.repr().int {
                    match integer {
                        IntegerType::Pointer(sign) => sign.abi_hash(tcx, hasher),
                        IntegerType::Fixed(integer, sign) => {
                            integer.int_ty_str().abi_hash(tcx, hasher);
                            sign.abi_hash(tcx, hasher);
                        }
                    }
                }

                if let Some(pack) = adt_def.repr().pack {
                    pack.bits().abi_hash(tcx, hasher);
                }

                adt_def.repr().c().abi_hash(tcx, hasher);

                for variant in adt_def.variants() {
                    variant.name.abi_hash(tcx, hasher);
                    for field in &variant.fields {
                        field.name.abi_hash(tcx, hasher);
                        let field_ty = tcx.type_of(field.did).instantiate_identity();
                        field_ty.abi_hash(tcx, hasher);
                    }
                }
                args.abi_hash(tcx, hasher);
            }

            ty::Tuple(args) if args.len() == 0 => {}

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
            | ty::Dynamic(_, _)
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
    fn abi_hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        for ty in self.inputs_and_output {
            ty.abi_hash(tcx, hasher);
        }
        self.safety.is_safe().abi_hash(tcx, hasher);
    }
}

impl<'tcx> AbiHashStable<'tcx> for ty::GenericArg<'tcx> {
    fn abi_hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        self.kind().abi_hash(tcx, hasher);
    }
}

impl<'tcx> AbiHashStable<'tcx> for ty::GenericArgKind<'tcx> {
    fn abi_hash(&self, tcx: TyCtxt<'tcx>, hasher: &mut StableHasher) {
        match self {
            ty::GenericArgKind::Type(t) => t.abi_hash(tcx, hasher),
            ty::GenericArgKind::Lifetime(_) | ty::GenericArgKind::Const(_) => unimplemented!(),
        }
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
        sig_ty.abi_hash(tcx, &mut hasher);
        hasher.finish::<Hash128>()
    };

    hash.as_u128().to_string()
}
