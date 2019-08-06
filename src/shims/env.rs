use rustc::ty::{layout::{Size, Align}, TyCtxt};
use rustc_mir::interpret::Memory;

use crate::*;

pub(crate) fn alloc_env_value<'mir, 'tcx>(bytes: &[u8], memory: &mut Memory<'mir, 'tcx, Evaluator<'tcx>>, tcx: &TyCtxt<'tcx>) -> Pointer<Tag> {
    let length = bytes.len() as u64;
    // `+1` for the null terminator.
    let ptr = memory.allocate(
        Size::from_bytes(length + 1),
        Align::from_bytes(1).unwrap(),
        MiriMemoryKind::Env.into(),
    );
    // We just allocated these, so the write cannot fail.
    let alloc = memory.get_mut(ptr.alloc_id).unwrap();
    alloc.write_bytes(tcx, ptr, &bytes).unwrap();
    let trailing_zero_ptr = ptr.offset(
        Size::from_bytes(length),
        tcx,
    ).unwrap();
    alloc.write_bytes(tcx, trailing_zero_ptr, &[0]).unwrap();
    ptr
}
