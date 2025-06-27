use std::ffi::{OsStr, OsString};
use std::io::ErrorKind;
use std::{env, mem};

use rustc_abi::{FieldIdx, Size};
use rustc_data_structures::fx::FxHashMap;
use rustc_index::IndexVec;
use rustc_middle::ty::Ty;

use crate::*;

pub struct UnixEnvVars<'tcx> {
    /// Stores pointers to the environment variables. These variables must be stored as
    /// null-terminated target strings (c_str or wide_str) with the `"{name}={value}"` format.
    map: FxHashMap<OsString, Pointer>,

    /// Place where the `environ` static is stored. Lazily initialized, but then never changes.
    environ: MPlaceTy<'tcx>,
}

impl VisitProvenance for UnixEnvVars<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        let UnixEnvVars { map, environ } = self;

        environ.visit_provenance(visit);
        for ptr in map.values() {
            ptr.visit_provenance(visit);
        }
    }
}

impl<'tcx> UnixEnvVars<'tcx> {
    pub(crate) fn new(
        ecx: &mut InterpCx<'tcx, MiriMachine<'tcx>>,
        env_vars: FxHashMap<OsString, OsString>,
    ) -> InterpResult<'tcx, Self> {
        // Allocate memory for all these env vars.
        let mut env_vars_machine = FxHashMap::default();
        for (name, val) in env_vars.into_iter() {
            let ptr = alloc_env_var(ecx, &name, &val)?;
            env_vars_machine.insert(name, ptr);
        }

        // This is memory backing an extern static, hence `ExternStatic`, not `Env`.
        let layout = ecx.machine.layouts.mut_raw_ptr;
        let environ = ecx.allocate(layout, MiriMemoryKind::ExternStatic.into())?;
        let environ_block = alloc_environ_block(ecx, env_vars_machine.values().copied().collect())?;
        ecx.write_pointer(environ_block, &environ)?;

        interp_ok(UnixEnvVars { map: env_vars_machine, environ })
    }

    pub(crate) fn cleanup(ecx: &mut InterpCx<'tcx, MiriMachine<'tcx>>) -> InterpResult<'tcx> {
        // Deallocate individual env vars.
        let env_vars = mem::take(&mut ecx.machine.env_vars.unix_mut().map);
        for (_name, ptr) in env_vars {
            ecx.deallocate_ptr(ptr, None, MiriMemoryKind::Runtime.into())?;
        }
        // Deallocate environ var list.
        let environ = &ecx.machine.env_vars.unix().environ;
        let old_vars_ptr = ecx.read_pointer(environ)?;
        ecx.deallocate_ptr(old_vars_ptr, None, MiriMemoryKind::Runtime.into())?;

        interp_ok(())
    }

    pub(crate) fn environ(&self) -> Pointer {
        self.environ.ptr()
    }

    fn get_ptr(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        name: &OsStr,
    ) -> InterpResult<'tcx, Option<Pointer>> {
        // We don't care about the value as we have the `map` to keep track of everything,
        // but we do want to do this read so it shows up as a data race.
        let _vars_ptr = ecx.read_pointer(&self.environ)?;
        let Some(var_ptr) = self.map.get(name) else {
            return interp_ok(None);
        };
        // The offset is used to strip the "{name}=" part of the string.
        let var_ptr = var_ptr.wrapping_offset(
            Size::from_bytes(u64::try_from(name.len()).unwrap().strict_add(1)),
            ecx,
        );
        interp_ok(Some(var_ptr))
    }

    /// Implementation detail for [`InterpCx::get_env_var`]. This basically does `getenv`, complete
    /// with the reads of the environment, but returns an [`OsString`] instead of a pointer.
    pub(crate) fn get(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        name: &OsStr,
    ) -> InterpResult<'tcx, Option<OsString>> {
        let var_ptr = self.get_ptr(ecx, name)?;
        if let Some(ptr) = var_ptr {
            let var = ecx.read_os_str_from_c_str(ptr)?;
            interp_ok(Some(var.to_owned()))
        } else {
            interp_ok(None)
        }
    }
}

fn alloc_env_var<'tcx>(
    ecx: &mut InterpCx<'tcx, MiriMachine<'tcx>>,
    name: &OsStr,
    value: &OsStr,
) -> InterpResult<'tcx, Pointer> {
    let mut name_osstring = name.to_os_string();
    name_osstring.push("=");
    name_osstring.push(value);
    ecx.alloc_os_str_as_c_str(name_osstring.as_os_str(), MiriMemoryKind::Runtime.into())
}

/// Allocates an `environ` block with the given list of pointers.
fn alloc_environ_block<'tcx>(
    ecx: &mut InterpCx<'tcx, MiriMachine<'tcx>>,
    mut vars: IndexVec<FieldIdx, Pointer>,
) -> InterpResult<'tcx, Pointer> {
    // Add trailing null.
    vars.push(Pointer::null());
    // Make an array with all these pointers inside Miri.
    let vars_layout = ecx.layout_of(Ty::new_array(
        *ecx.tcx,
        ecx.machine.layouts.mut_raw_ptr.ty,
        u64::try_from(vars.len()).unwrap(),
    ))?;
    let vars_place = ecx.allocate(vars_layout, MiriMemoryKind::Runtime.into())?;
    for (idx, var) in vars.into_iter_enumerated() {
        let place = ecx.project_field(&vars_place, idx)?;
        ecx.write_pointer(var, &place)?;
    }
    interp_ok(vars_place.ptr())
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn getenv(&mut self, name_op: &OpTy<'tcx>) -> InterpResult<'tcx, Pointer> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("getenv");

        let name_ptr = this.read_pointer(name_op)?;
        let name = this.read_os_str_from_c_str(name_ptr)?;

        let var_ptr = this.machine.env_vars.unix().get_ptr(this, name)?;
        interp_ok(var_ptr.unwrap_or_else(Pointer::null))
    }

    fn setenv(
        &mut self,
        name_op: &OpTy<'tcx>,
        value_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("setenv");

        let name_ptr = this.read_pointer(name_op)?;
        let value_ptr = this.read_pointer(value_op)?;

        let mut new = None;
        if !this.ptr_is_null(name_ptr)? {
            let name = this.read_os_str_from_c_str(name_ptr)?;
            if !name.is_empty() && !name.to_string_lossy().contains('=') {
                let value = this.read_os_str_from_c_str(value_ptr)?;
                new = Some((name.to_owned(), value.to_owned()));
            }
        }
        if let Some((name, value)) = new {
            let var_ptr = alloc_env_var(this, &name, &value)?;
            if let Some(var) = this.machine.env_vars.unix_mut().map.insert(name, var_ptr) {
                this.deallocate_ptr(var, None, MiriMemoryKind::Runtime.into())?;
            }
            this.update_environ()?;
            interp_ok(Scalar::from_i32(0)) // return zero on success
        } else {
            // name argument is a null pointer, points to an empty string, or points to a string containing an '=' character.
            this.set_last_error_and_return_i32(LibcError("EINVAL"))
        }
    }

    fn unsetenv(&mut self, name_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("unsetenv");

        let name_ptr = this.read_pointer(name_op)?;
        let mut success = None;
        if !this.ptr_is_null(name_ptr)? {
            let name = this.read_os_str_from_c_str(name_ptr)?.to_owned();
            if !name.is_empty() && !name.to_string_lossy().contains('=') {
                success = Some(this.machine.env_vars.unix_mut().map.remove(&name));
            }
        }
        if let Some(old) = success {
            if let Some(var) = old {
                this.deallocate_ptr(var, None, MiriMemoryKind::Runtime.into())?;
            }
            this.update_environ()?;
            interp_ok(Scalar::from_i32(0))
        } else {
            // name argument is a null pointer, points to an empty string, or points to a string containing an '=' character.
            this.set_last_error_and_return_i32(LibcError("EINVAL"))
        }
    }

    fn getcwd(&mut self, buf_op: &OpTy<'tcx>, size_op: &OpTy<'tcx>) -> InterpResult<'tcx, Pointer> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("getcwd");

        let buf = this.read_pointer(buf_op)?;
        let size = this.read_target_usize(size_op)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`getcwd`", reject_with)?;
            this.set_last_error(ErrorKind::PermissionDenied)?;
            return interp_ok(Pointer::null());
        }

        // If we cannot get the current directory, we return null
        match env::current_dir() {
            Ok(cwd) => {
                if this.write_path_to_c_str(&cwd, buf, size)?.0 {
                    return interp_ok(buf);
                }
                this.set_last_error(LibcError("ERANGE"))?;
            }
            Err(e) => this.set_last_error(e)?,
        }

        interp_ok(Pointer::null())
    }

    fn chdir(&mut self, path_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("chdir");

        let path = this.read_path_from_c_str(this.read_pointer(path_op)?)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`chdir`", reject_with)?;
            return this.set_last_error_and_return_i32(ErrorKind::PermissionDenied);
        }

        let result = env::set_current_dir(path).map(|()| 0);
        interp_ok(Scalar::from_i32(this.try_unwrap_io_result(result)?))
    }

    /// Updates the `environ` static.
    fn update_environ(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // Deallocate the old environ list.
        let environ = this.machine.env_vars.unix().environ.clone();
        let old_vars_ptr = this.read_pointer(&environ)?;
        this.deallocate_ptr(old_vars_ptr, None, MiriMemoryKind::Runtime.into())?;

        // Write the new list.
        let vals = this.machine.env_vars.unix().map.values().copied().collect();
        let environ_block = alloc_environ_block(this, vals)?;
        this.write_pointer(environ_block, &environ)?;

        interp_ok(())
    }

    fn getpid(&mut self) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("getpid");

        // The reason we need to do this wacky of a conversion is because
        // `libc::getpid` returns an i32, however, `std::process::id()` return an u32.
        // So we un-do the conversion that stdlib does and turn it back into an i32.
        // In `Scalar` representation, these are the same, so we don't need to anything else.
        interp_ok(Scalar::from_u32(this.get_pid()))
    }

    fn linux_gettid(&mut self) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_ref();
        this.assert_target_os("linux", "gettid");

        let index = this.machine.threads.active_thread().to_u32();

        // Compute a TID for this thread, ensuring that the main thread has PID == TID.
        let tid = this.get_pid().strict_add(index);

        interp_ok(Scalar::from_u32(tid))
    }
}
