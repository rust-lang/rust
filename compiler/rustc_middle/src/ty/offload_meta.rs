use crate::ty::{self, PseudoCanonicalInput, Ty, TyCtxt, TypingEnv};

// TODO(Sa4dUs): it doesn't feel correct for me to place this on `rustc_ast::expand`, will look for a proper location
pub struct OffloadMetadata {
    pub payload_size: u64,
    pub mode: TransferKind,
}

pub enum TransferKind {
    FromGpu = 1,
    ToGpu = 2,
    Both = 3,
}

impl OffloadMetadata {
    pub fn new(payload_size: u64, mode: TransferKind) -> Self {
        OffloadMetadata { payload_size, mode }
    }

    pub fn from_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        OffloadMetadata { payload_size: get_payload_size(tcx, ty), mode: TransferKind::Both }
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
