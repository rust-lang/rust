use std::collections::HashMap;

use rustc::ty::{layout::{Size, Align}, TyCtxt};
use rustc_mir::interpret::{Pointer, Memory};
use crate::stacked_borrows::Tag;
use crate::*;

#[derive(Default)]
pub struct EnvVars {
    map: HashMap<Vec<u8>, Pointer<Tag>>,
}

impl EnvVars {
    pub(crate) fn init<'mir, 'tcx>(
        ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
        tcx: &TyCtxt<'tcx>,
        communicate: bool,
    ) {
        if communicate {
            for (name, value) in std::env::vars() {
                let value = alloc_env_value(value.as_bytes(), ecx.memory_mut(), tcx);
                ecx.machine.env_vars.map.insert(name.into_bytes(), value);
            }
        }
    }

    pub(crate) fn get(&self, name: &[u8]) -> Option<&Pointer<Tag>> {
        self.map.get(name)
    }

    pub(crate) fn unset(&mut self, name: &[u8]) -> Option<Pointer<Tag>> {
        self.map.remove(name)
    }

    pub(crate) fn set(&mut self, name: Vec<u8>, ptr: Pointer<Tag>) -> Option<Pointer<Tag>>{
        self.map.insert(name, ptr)
    }
}

pub(crate) fn alloc_env_value<'mir, 'tcx>(
    bytes: &[u8],
    memory: &mut Memory<'mir, 'tcx, Evaluator<'tcx>>,
    tcx: &TyCtxt<'tcx>,
) -> Pointer<Tag> {
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
