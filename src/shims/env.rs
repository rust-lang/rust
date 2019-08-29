use std::collections::HashMap;

use rustc::ty::layout::{Size, Align};
use rustc_mir::interpret::{Pointer, Memory};
use crate::stacked_borrows::Tag;
use crate::*;

#[derive(Default)]
pub struct EnvVars {
    /// Stores pointers to the environment variables. These variables must be stored as
    /// null-terminated C strings with the `"{name}={value}"` format.
    map: HashMap<Vec<u8>, Pointer<Tag>>,
}

impl EnvVars {
    pub(crate) fn init<'mir, 'tcx>(
        ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
        excluded_env_vars: Vec<String>,
    ) {
        if ecx.machine.communicate {
            for (name, value) in std::env::vars() {
                if !excluded_env_vars.contains(&name) {
                    let var_ptr = alloc_env_var(name.as_bytes(), value.as_bytes(), ecx.memory_mut());
                    ecx.machine.env_vars.map.insert(name.into_bytes(), var_ptr);
                }
            }
        }
    }
}

fn alloc_env_var<'mir, 'tcx>(
    name: &[u8],
    value: &[u8],
    memory: &mut Memory<'mir, 'tcx, Evaluator<'tcx>>,
) -> Pointer<Tag> {
    let bytes = [name, b"=", value].concat();
    let tcx = {memory.tcx.tcx};
    let length = bytes.len() as u64;
    // `+1` for the null terminator.
    let ptr = memory.allocate(
        Size::from_bytes(length + 1),
        Align::from_bytes(1).unwrap(),
        MiriMemoryKind::Env.into(),
    );
    // We just allocated these, so the write cannot fail.
    let alloc = memory.get_mut(ptr.alloc_id).unwrap();
    alloc.write_bytes(&tcx, ptr, &bytes).unwrap();
    let trailing_zero_ptr = ptr.offset(
        Size::from_bytes(length),
        &tcx,
    ).unwrap();
    alloc.write_bytes(&tcx, trailing_zero_ptr, &[0]).unwrap();
    ptr
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn getenv(
        &mut self,
        name_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let name = this.memory().read_c_str(name_ptr)?;
        Ok(match this.machine.env_vars.map.get(name) {
            // The offset is used to strip the "{name}=" part of the string.
            Some(var_ptr) => Scalar::Ptr(var_ptr.offset(Size::from_bytes(name.len() as u64 + 1), this)?),
            None => Scalar::ptr_null(&*this.tcx),
        })
    }

    fn setenv(
        &mut self,
        name_op: OpTy<'tcx, Tag>,
        value_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let value_ptr = this.read_scalar(value_op)?.not_undef()?;
        let value = this.memory().read_c_str(value_ptr)?;
        let mut new = None;
        if !this.is_null(name_ptr)? {
            let name = this.memory().read_c_str(name_ptr)?;
            if !name.is_empty() && !name.contains(&b'=') {
                new = Some((name.to_owned(), value.to_owned()));
            }
        }
        if let Some((name, value)) = new {
            let var_ptr = alloc_env_var(&name, &value, this.memory_mut());
            if let Some(var) = this.machine.env_vars.map.insert(name.to_owned(), var_ptr) {
                this.memory_mut().deallocate(var, None, MiriMemoryKind::Env.into())?;
            }
            Ok(0)
        } else {
            Ok(-1)
        }
    }

    fn unsetenv(
        &mut self,
        name_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let mut success = None;
        if !this.is_null(name_ptr)? {
            let name = this.memory().read_c_str(name_ptr)?.to_owned();
            if !name.is_empty() && !name.contains(&b'=') {
                success = Some(this.machine.env_vars.map.remove(&name));
            }
        }
        if let Some(old) = success {
            if let Some(var) = old {
                this.memory_mut().deallocate(var, None, MiriMemoryKind::Env.into())?;
            }
            Ok(0)
        } else {
            Ok(-1)
        }
    }
}
