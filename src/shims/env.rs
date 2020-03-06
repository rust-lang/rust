use std::ffi::{OsString, OsStr};
use std::env;

use crate::stacked_borrows::Tag;
use crate::rustc_target::abi::LayoutOf;
use crate::*;

use rustc_data_structures::fx::FxHashMap;
use rustc::ty::layout::Size;
use rustc_mir::interpret::Pointer;

#[derive(Default)]
pub struct EnvVars {
    /// Stores pointers to the environment variables. These variables must be stored as
    /// null-terminated C strings with the `"{name}={value}"` format.
    map: FxHashMap<OsString, Pointer<Tag>>,
}

impl EnvVars {
    pub(crate) fn init<'mir, 'tcx>(
        ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
        excluded_env_vars: Vec<String>,
    ) {
        if ecx.machine.communicate {
            for (name, value) in env::vars() {
                if !excluded_env_vars.contains(&name) {
                    let var_ptr =
                        alloc_env_var_as_c_str(name.as_ref(), value.as_ref(), ecx);
                    ecx.machine.env_vars.map.insert(OsString::from(name), var_ptr);
                }
            }
        }
        ecx.update_environ().unwrap();
    }
}

fn alloc_env_var_as_c_str<'mir, 'tcx>(
    name: &OsStr,
    value: &OsStr,
    ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
) -> Pointer<Tag> {
    let mut name_osstring = name.to_os_string();
    name_osstring.push("=");
    name_osstring.push(value);
    ecx.alloc_os_str_as_c_str(name_osstring.as_os_str(), MiriMemoryKind::Machine.into())
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn getenv(&mut self, name_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let name = this.read_os_str_from_c_str(name_ptr)?;
        Ok(match this.machine.env_vars.map.get(name) {
            // The offset is used to strip the "{name}=" part of the string.
            Some(var_ptr) => {
                Scalar::from(var_ptr.offset(Size::from_bytes(name.len() as u64 + 1), this)?)
            }
            None => Scalar::ptr_null(&*this.tcx),
        })
    }

    fn setenv(
        &mut self,
        name_op: OpTy<'tcx, Tag>,
        value_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let mut this = self.eval_context_mut();

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let value_ptr = this.read_scalar(value_op)?.not_undef()?;
        let value = this.read_os_str_from_c_str(value_ptr)?;
        let mut new = None;
        if !this.is_null(name_ptr)? {
            let name = this.read_os_str_from_c_str(name_ptr)?;
            if !name.is_empty() && !name.to_string_lossy().contains('=') {
                new = Some((name.to_owned(), value.to_owned()));
            }
        }
        if let Some((name, value)) = new {
            let var_ptr = alloc_env_var_as_c_str(&name, &value, &mut this);
            if let Some(var) = this.machine.env_vars.map.insert(name.to_owned(), var_ptr) {
                this.memory
                    .deallocate(var, None, MiriMemoryKind::Machine.into())?;
            }
            this.update_environ()?;
            Ok(0)
        } else {
            Ok(-1)
        }
    }

    fn unsetenv(&mut self, name_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let mut success = None;
        if !this.is_null(name_ptr)? {
            let name = this.read_os_str_from_c_str(name_ptr)?.to_owned();
            if !name.is_empty() && !name.to_string_lossy().contains('=') {
                success = Some(this.machine.env_vars.map.remove(&name));
            }
        }
        if let Some(old) = success {
            if let Some(var) = old {
                this.memory
                    .deallocate(var, None, MiriMemoryKind::Machine.into())?;
            }
            this.update_environ()?;
            Ok(0)
        } else {
            Ok(-1)
        }
    }

    fn getcwd(
        &mut self,
        buf_op: OpTy<'tcx, Tag>,
        size_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();

        this.check_no_isolation("getcwd")?;

        let buf = this.read_scalar(buf_op)?.not_undef()?;
        let size = this.read_scalar(size_op)?.to_machine_usize(&*this.tcx)?;
        // If we cannot get the current directory, we return null
        match env::current_dir() {
            Ok(cwd) => {
                if this.write_os_str_to_c_str(&OsString::from(cwd), buf, size)?.0 {
                    return Ok(buf);
                }
                let erange = this.eval_libc("ERANGE")?;
                this.set_last_error(erange)?;
            }
            Err(e) => this.set_last_error_from_io_error(e)?,
        }
        Ok(Scalar::ptr_null(&*this.tcx))
    }

    fn chdir(&mut self, path_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("chdir")?;

        let path = this.read_os_str_from_c_str(this.read_scalar(path_op)?.not_undef()?)?;

        match env::set_current_dir(path) {
            Ok(()) => Ok(0),
            Err(e) => {
                this.set_last_error_from_io_error(e)?;
                Ok(-1)
            }
        }
    }

    /// Updates the `environ` static. It should not be called before
    /// `MemoryExtra::init_extern_statics`.
    fn update_environ(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // Deallocate the old environ value.
        let old_vars_ptr = this.read_scalar(this.memory.extra.environ.unwrap().into())?.not_undef()?;
        // The pointer itself can be null because `MemoryExtra::init_extern_statics` only
        // initializes the place for the static but not the static itself.
        if !this.is_null(old_vars_ptr)? {
            this.memory.deallocate(this.force_ptr(old_vars_ptr)?, None, MiriMemoryKind::Machine.into())?;
        }
        // Collect all the pointers to each variable in a vector.
        let mut vars: Vec<Scalar<Tag>> = this.machine.env_vars.map.values().map(|&ptr| ptr.into()).collect();
        // Add the trailing null pointer.
        vars.push(Scalar::from_int(0, this.pointer_size()));
        // Make an array with all these pointers inside Miri.
        let tcx = this.tcx;
        let vars_layout =
            this.layout_of(tcx.mk_array(tcx.types.usize, vars.len() as u64))?;
        let vars_place = this.allocate(vars_layout, MiriMemoryKind::Machine.into());
        for (idx, var) in vars.into_iter().enumerate() {
            let place = this.mplace_field(vars_place, idx as u64)?;
            this.write_scalar(var, place.into())?;
        }
        this.write_scalar(
            vars_place.ptr,
            this.memory.extra.environ.unwrap().into(),
        )?;

        Ok(())
    }
}
