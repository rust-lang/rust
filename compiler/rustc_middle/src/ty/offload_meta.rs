use crate::ty::{self, PseudoCanonicalInput, Ty, TyCtxt, TypingEnv};

// TODO(Sa4dUs): it doesn't feel correct for me to place this on `rustc_ast::expand`, will look for a proper location
pub struct OffloadMetadata {
    pub payload_size: u64,
    pub mode: TransferKind,
}

// TODO(Sa4dUs): add `OMP_MAP_TARGET_PARAM = 0x20` flag only when needed
#[repr(u64)]
#[derive(Debug, Copy, Clone)]
pub enum TransferKind {
    FromGpu = 1,
    ToGpu = 2,
    Both = 1 + 2,
}

impl OffloadMetadata {
    pub fn new(payload_size: u64, mode: TransferKind) -> Self {
        OffloadMetadata { payload_size, mode }
    }

    pub fn from_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        OffloadMetadata {
            payload_size: get_payload_size(tcx, ty),
            mode: TransferKind::from_ty(tcx, ty),
        }
    }
}

// TODO(Sa4dUs): WIP, rn we just have a naive logic for references
fn get_payload_size<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> u64 {
    match ty.kind() {
        ty::RawPtr(inner, _) | ty::Ref(_, inner, _) => get_payload_size(tcx, *inner),
        _ => tcx
            .layout_of(PseudoCanonicalInput {
                typing_env: TypingEnv::fully_monomorphized(),
                value: ty,
            })
            .unwrap()
            .size
            .bytes(),
    }
}

impl TransferKind {
    pub fn from_ty<'tcx>(_tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        // TODO(Sa4dUs): this logic is probs not fully correct, but it works for now
        match ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_) => TransferKind::ToGpu,

            ty::Adt(_, _)
            | ty::Tuple(_)
            | ty::Array(_, _) => TransferKind::ToGpu,

            ty::RawPtr(_, rustc_ast::Mutability::Not)
            | ty::Ref(_, _, rustc_ast::Mutability::Not) => TransferKind::ToGpu,

            ty::RawPtr(_, rustc_ast::Mutability::Mut)
            | ty::Ref(_, _, rustc_ast::Mutability::Mut) => TransferKind::Both,

            ty::Slice(_)
            | ty::Str
            | ty::Dynamic(_, _) => TransferKind::Both,

            ty::FnDef(_, _)
            | ty::FnPtr(_, _)
            | ty::Closure(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(_, _) => TransferKind::ToGpu,

            ty::Alias(_, _)
            | ty::Param(_)
            | ty::Bound(_, _)
            | ty::Placeholder(_)
            | ty::Infer(_)
            | ty::Error(_) => TransferKind::ToGpu,

            ty::Never => TransferKind::ToGpu,
            ty::Foreign(_) => TransferKind::Both,
            ty::Pat(_, _) => TransferKind::Both,
            ty::UnsafeBinder(_) => TransferKind::Both,
        }
    }
}
