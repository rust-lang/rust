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
        /*
        rustc_middle::infer::canonical::ir::TyKind::Bool => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Char => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Int(int_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Uint(uint_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Float(float_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Adt(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Foreign(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Str => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Array(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Pat(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Slice(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::RawPtr(_, mutability) => todo!(),
        */
        ty::Ref(_, inner, _) => get_payload_size(tcx, *inner),
        /*
        rustc_middle::infer::canonical::ir::TyKind::FnDef(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::FnPtr(binder, fn_header) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::UnsafeBinder(unsafe_binder_inner) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Dynamic(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Closure(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::CoroutineClosure(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Coroutine(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::CoroutineWitness(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Never => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Tuple(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Alias(alias_ty_kind, alias_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Param(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Bound(bound_var_index_kind, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Placeholder(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Infer(infer_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Error(_) => todo!(),
        */
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
            rustc_type_ir::TyKind::Bool
            | rustc_type_ir::TyKind::Char
            | rustc_type_ir::TyKind::Int(_)
            | rustc_type_ir::TyKind::Uint(_)
            | rustc_type_ir::TyKind::Float(_) => TransferKind::ToGpu,

            rustc_type_ir::TyKind::Adt(_, _)
            | rustc_type_ir::TyKind::Tuple(_)
            | rustc_type_ir::TyKind::Array(_, _) => TransferKind::ToGpu,

            rustc_type_ir::TyKind::RawPtr(_, rustc_ast::Mutability::Not)
            | rustc_type_ir::TyKind::Ref(_, _, rustc_ast::Mutability::Not) => TransferKind::ToGpu,

            rustc_type_ir::TyKind::RawPtr(_, rustc_ast::Mutability::Mut)
            | rustc_type_ir::TyKind::Ref(_, _, rustc_ast::Mutability::Mut) => TransferKind::Both,

            rustc_type_ir::TyKind::Slice(_)
            | rustc_type_ir::TyKind::Str
            | rustc_type_ir::TyKind::Dynamic(_, _) => TransferKind::Both,

            rustc_type_ir::TyKind::FnDef(_, _)
            | rustc_type_ir::TyKind::FnPtr(_, _)
            | rustc_type_ir::TyKind::Closure(_, _)
            | rustc_type_ir::TyKind::CoroutineClosure(_, _)
            | rustc_type_ir::TyKind::Coroutine(_, _)
            | rustc_type_ir::TyKind::CoroutineWitness(_, _) => TransferKind::ToGpu,

            rustc_type_ir::TyKind::Alias(_, _)
            | rustc_type_ir::TyKind::Param(_)
            | rustc_type_ir::TyKind::Bound(_, _)
            | rustc_type_ir::TyKind::Placeholder(_)
            | rustc_type_ir::TyKind::Infer(_)
            | rustc_type_ir::TyKind::Error(_) => TransferKind::ToGpu,

            rustc_type_ir::TyKind::Never => TransferKind::ToGpu,
            rustc_type_ir::TyKind::Foreign(_) => TransferKind::Both,
            rustc_type_ir::TyKind::Pat(_, _) => TransferKind::Both,
            rustc_type_ir::TyKind::UnsafeBinder(_) => TransferKind::Both,
        }
    }
}
