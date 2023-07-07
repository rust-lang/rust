use std::env;
use std::ffi::{OsStr, OsString};
use std::io::ErrorKind;
use std::mem;

use rustc_const_eval::interpret::Pointer;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::Ty;
use rustc_target::abi::Size;

use crate::helpers::target_os_is_unix;
use crate::*;

/// Check whether an operation that writes to a target buffer was successful.
/// Accordingly select return value.
/// Local helper function to be used in Windows shims.
fn windows_check_buffer_size((success, len): (bool, u64)) -> u32 {
    if success {
        // If the function succeeds, the return value is the number of characters stored in the target buffer,
        // not including the terminating null character.
        u32::try_from(len.checked_sub(1).unwrap()).unwrap()
    } else {
        // If the target buffer was not large enough to hold the data, the return value is the buffer size, in characters,
        // required to hold the string and its terminating null character.
        u32::try_from(len).unwrap()
    }
}

#[derive(Default)]
pub struct EnvVars<'tcx> {
    /// Stores pointers to the environment variables. These variables must be stored as
    /// null-terminated target strings (c_str or wide_str) with the `"{name}={value}"` format.
    map: FxHashMap<OsString, Pointer<Option<Provenance>>>,

    /// Place where the `environ` static is stored. Lazily initialized, but then never changes.
    pub(crate) environ: Option<MPlaceTy<'tcx, Provenance>>,
}

impl VisitTags for EnvVars<'_> {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        let EnvVars { map, environ } = self;

        environ.visit_tags(visit);
        for ptr in map.values() {
            ptr.visit_tags(visit);
        }
    }
}

impl<'tcx> EnvVars<'tcx> {
    pub(crate) fn init<'mir>(
        ecx: &mut InterpCx<'mir, 'tcx, MiriMachine<'mir, 'tcx>>,
        config: &MiriConfig,
    ) -> InterpResult<'tcx> {
        let target_os = ecx.tcx.sess.target.os.as_ref();

        // Skip the loop entirely if we don't want to forward anything.
        if ecx.machine.communicate() || !config.forwarded_env_vars.is_empty() {
            for (name, value) in &config.env {
                let forward = ecx.machine.communicate()
                    || config.forwarded_env_vars.iter().any(|v| **v == *name);
                if forward {
                    let var_ptr = match target_os {
                        target if target_os_is_unix(target) =>
                            alloc_env_var_as_c_str(name.as_ref(), value.as_ref(), ecx)?,
                        "windows" => alloc_env_var_as_wide_str(name.as_ref(), value.as_ref(), ecx)?,
                        unsupported =>
                            throw_unsup_format!(
                                "environment support for target OS `{}` not yet available",
                                unsupported
                            ),
                    };
                    ecx.machine.env_vars.map.insert(name.clone(), var_ptr);
                }
            }
        }
        ecx.update_environ()
    }

    pub(crate) fn cleanup<'mir>(
        ecx: &mut InterpCx<'mir, 'tcx, MiriMachine<'mir, 'tcx>>,
    ) -> InterpResult<'tcx> {
        // Deallocate individual env vars.
        let env_vars = mem::take(&mut ecx.machine.env_vars.map);
        for (_name, ptr) in env_vars {
            ecx.deallocate_ptr(ptr, None, MiriMemoryKind::Runtime.into())?;
        }
        // Deallocate environ var list.
        let environ = ecx.machine.env_vars.environ.unwrap();
        let old_vars_ptr = ecx.read_pointer(&environ.into())?;
        ecx.deallocate_ptr(old_vars_ptr, None, MiriMemoryKind::Runtime.into())?;
        Ok(())
    }
}

fn alloc_env_var_as_c_str<'mir, 'tcx>(
    name: &OsStr,
    value: &OsStr,
    ecx: &mut InterpCx<'mir, 'tcx, MiriMachine<'mir, 'tcx>>,
) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
    let mut name_osstring = name.to_os_string();
    name_osstring.push("=");
    name_osstring.push(value);
    ecx.alloc_os_str_as_c_str(name_osstring.as_os_str(), MiriMemoryKind::Runtime.into())
}

fn alloc_env_var_as_wide_str<'mir, 'tcx>(
    name: &OsStr,
    value: &OsStr,
    ecx: &mut InterpCx<'mir, 'tcx, MiriMachine<'mir, 'tcx>>,
) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
    let mut name_osstring = name.to_os_string();
    name_osstring.push("=");
    name_osstring.push(value);
    ecx.alloc_os_str_as_wide_str(name_osstring.as_os_str(), MiriMemoryKind::Runtime.into())
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn getenv(
        &mut self,
        name_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("getenv");

        let name_ptr = this.read_pointer(name_op)?;
        let name = this.read_os_str_from_c_str(name_ptr)?;
        Ok(match this.machine.env_vars.map.get(name) {
            Some(var_ptr) => {
                // The offset is used to strip the "{name}=" part of the string.
                var_ptr.offset(
                    Size::from_bytes(u64::try_from(name.len()).unwrap().checked_add(1).unwrap()),
                    this,
                )?
            }
            None => Pointer::null(),
        })
    }

    #[allow(non_snake_case)]
    fn GetEnvironmentVariableW(
        &mut self,
        name_op: &OpTy<'tcx, Provenance>, // LPCWSTR
        buf_op: &OpTy<'tcx, Provenance>,  // LPWSTR
        size_op: &OpTy<'tcx, Provenance>, // DWORD
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        // ^ Returns DWORD (u32 on Windows)

        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetEnvironmentVariableW");

        let name_ptr = this.read_pointer(name_op)?;
        let name = this.read_os_str_from_wide_str(name_ptr)?;
        Ok(match this.machine.env_vars.map.get(&name) {
            Some(var_ptr) => {
                // The offset is used to strip the "{name}=" part of the string.
                #[rustfmt::skip]
                let name_offset_bytes = u64::try_from(name.len()).unwrap()
                    .checked_add(1).unwrap()
                    .checked_mul(2).unwrap();
                let var_ptr = var_ptr.offset(Size::from_bytes(name_offset_bytes), this)?;
                let var = this.read_os_str_from_wide_str(var_ptr)?;

                let buf_ptr = this.read_pointer(buf_op)?;
                // `buf_size` represents the size in characters.
                let buf_size = u64::from(this.read_scalar(size_op)?.to_u32()?);
                Scalar::from_u32(windows_check_buffer_size(
                    this.write_os_str_to_wide_str(
                        &var, buf_ptr, buf_size, /*truncate*/ false,
                    )?,
                ))
            }
            None => {
                let envvar_not_found = this.eval_windows("c", "ERROR_ENVVAR_NOT_FOUND");
                this.set_last_error(envvar_not_found)?;
                Scalar::from_u32(0) // return zero upon failure
            }
        })
    }

    #[allow(non_snake_case)]
    fn GetEnvironmentStringsW(&mut self) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetEnvironmentStringsW");

        // Info on layout of environment blocks in Windows:
        // https://docs.microsoft.com/en-us/windows/win32/procthread/environment-variables
        let mut env_vars = std::ffi::OsString::new();
        for &item in this.machine.env_vars.map.values() {
            let env_var = this.read_os_str_from_wide_str(item)?;
            env_vars.push(env_var);
            env_vars.push("\0");
        }
        // Allocate environment block & Store environment variables to environment block.
        // Final null terminator(block terminator) is added by `alloc_os_str_to_wide_str`.
        let envblock_ptr =
            this.alloc_os_str_as_wide_str(&env_vars, MiriMemoryKind::Runtime.into())?;
        // If the function succeeds, the return value is a pointer to the environment block of the current process.
        Ok(envblock_ptr)
    }

    #[allow(non_snake_case)]
    fn FreeEnvironmentStringsW(
        &mut self,
        env_block_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "FreeEnvironmentStringsW");

        let env_block_ptr = this.read_pointer(env_block_op)?;
        let result = this.deallocate_ptr(env_block_ptr, None, MiriMemoryKind::Runtime.into());
        // If the function succeeds, the return value is nonzero.
        Ok(Scalar::from_i32(i32::from(result.is_ok())))
    }

    fn setenv(
        &mut self,
        name_op: &OpTy<'tcx, Provenance>,
        value_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
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
            let var_ptr = alloc_env_var_as_c_str(&name, &value, this)?;
            if let Some(var) = this.machine.env_vars.map.insert(name, var_ptr) {
                this.deallocate_ptr(var, None, MiriMemoryKind::Runtime.into())?;
            }
            this.update_environ()?;
            Ok(0) // return zero on success
        } else {
            // name argument is a null pointer, points to an empty string, or points to a string containing an '=' character.
            let einval = this.eval_libc("EINVAL");
            this.set_last_error(einval)?;
            Ok(-1)
        }
    }

    #[allow(non_snake_case)]
    fn SetEnvironmentVariableW(
        &mut self,
        name_op: &OpTy<'tcx, Provenance>,  // LPCWSTR
        value_op: &OpTy<'tcx, Provenance>, // LPCWSTR
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "SetEnvironmentVariableW");

        let name_ptr = this.read_pointer(name_op)?;
        let value_ptr = this.read_pointer(value_op)?;

        if this.ptr_is_null(name_ptr)? {
            // ERROR CODE is not clearly explained in docs.. For now, throw UB instead.
            throw_ub_format!("pointer to environment variable name is NULL");
        }

        let name = this.read_os_str_from_wide_str(name_ptr)?;
        if name.is_empty() {
            throw_unsup_format!("environment variable name is an empty string");
        } else if name.to_string_lossy().contains('=') {
            throw_unsup_format!("environment variable name contains '='");
        } else if this.ptr_is_null(value_ptr)? {
            // Delete environment variable `{name}`
            if let Some(var) = this.machine.env_vars.map.remove(&name) {
                this.deallocate_ptr(var, None, MiriMemoryKind::Runtime.into())?;
                this.update_environ()?;
            }
            Ok(this.eval_windows("c", "TRUE"))
        } else {
            let value = this.read_os_str_from_wide_str(value_ptr)?;
            let var_ptr = alloc_env_var_as_wide_str(&name, &value, this)?;
            if let Some(var) = this.machine.env_vars.map.insert(name, var_ptr) {
                this.deallocate_ptr(var, None, MiriMemoryKind::Runtime.into())?;
            }
            this.update_environ()?;
            Ok(this.eval_windows("c", "TRUE"))
        }
    }

    fn unsetenv(&mut self, name_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("unsetenv");

        let name_ptr = this.read_pointer(name_op)?;
        let mut success = None;
        if !this.ptr_is_null(name_ptr)? {
            let name = this.read_os_str_from_c_str(name_ptr)?.to_owned();
            if !name.is_empty() && !name.to_string_lossy().contains('=') {
                success = Some(this.machine.env_vars.map.remove(&name));
            }
        }
        if let Some(old) = success {
            if let Some(var) = old {
                this.deallocate_ptr(var, None, MiriMemoryKind::Runtime.into())?;
            }
            this.update_environ()?;
            Ok(0)
        } else {
            // name argument is a null pointer, points to an empty string, or points to a string containing an '=' character.
            let einval = this.eval_libc("EINVAL");
            this.set_last_error(einval)?;
            Ok(-1)
        }
    }

    fn getcwd(
        &mut self,
        buf_op: &OpTy<'tcx, Provenance>,
        size_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Pointer<Option<Provenance>>> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("getcwd");

        let buf = this.read_pointer(buf_op)?;
        let size = this.read_target_usize(size_op)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`getcwd`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(Pointer::null());
        }

        // If we cannot get the current directory, we return null
        match env::current_dir() {
            Ok(cwd) => {
                if this.write_path_to_c_str(&cwd, buf, size)?.0 {
                    return Ok(buf);
                }
                let erange = this.eval_libc("ERANGE");
                this.set_last_error(erange)?;
            }
            Err(e) => this.set_last_error_from_io_error(e.kind())?,
        }

        Ok(Pointer::null())
    }

    #[allow(non_snake_case)]
    fn GetCurrentDirectoryW(
        &mut self,
        size_op: &OpTy<'tcx, Provenance>, // DWORD
        buf_op: &OpTy<'tcx, Provenance>,  // LPTSTR
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetCurrentDirectoryW");

        let size = u64::from(this.read_scalar(size_op)?.to_u32()?);
        let buf = this.read_pointer(buf_op)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`GetCurrentDirectoryW`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(Scalar::from_u32(0));
        }

        // If we cannot get the current directory, we return 0
        match env::current_dir() {
            Ok(cwd) =>
                return Ok(Scalar::from_u32(windows_check_buffer_size(
                    this.write_path_to_wide_str(&cwd, buf, size, /*truncate*/ false)?,
                ))),
            Err(e) => this.set_last_error_from_io_error(e.kind())?,
        }
        Ok(Scalar::from_u32(0))
    }

    fn chdir(&mut self, path_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("chdir");

        let path = this.read_path_from_c_str(this.read_pointer(path_op)?)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`chdir`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;

            return Ok(-1);
        }

        match env::set_current_dir(path) {
            Ok(()) => Ok(0),
            Err(e) => {
                this.set_last_error_from_io_error(e.kind())?;
                Ok(-1)
            }
        }
    }

    #[allow(non_snake_case)]
    fn SetCurrentDirectoryW(
        &mut self,
        path_op: &OpTy<'tcx, Provenance>, // LPCTSTR
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        // ^ Returns BOOL (i32 on Windows)

        let this = self.eval_context_mut();
        this.assert_target_os("windows", "SetCurrentDirectoryW");

        let path = this.read_path_from_wide_str(this.read_pointer(path_op)?)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`SetCurrentDirectoryW`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;

            return Ok(this.eval_windows("c", "FALSE"));
        }

        match env::set_current_dir(path) {
            Ok(()) => Ok(this.eval_windows("c", "TRUE")),
            Err(e) => {
                this.set_last_error_from_io_error(e.kind())?;
                Ok(this.eval_windows("c", "FALSE"))
            }
        }
    }

    /// Updates the `environ` static.
    /// The first time it gets called, also initializes `extra.environ`.
    fn update_environ(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // Deallocate the old environ list, if any.
        if let Some(environ) = this.machine.env_vars.environ {
            let old_vars_ptr = this.read_pointer(&environ.into())?;
            this.deallocate_ptr(old_vars_ptr, None, MiriMemoryKind::Runtime.into())?;
        } else {
            // No `environ` allocated yet, let's do that.
            // This is memory backing an extern static, hence `ExternStatic`, not `Env`.
            let layout = this.machine.layouts.mut_raw_ptr;
            let place = this.allocate(layout, MiriMemoryKind::ExternStatic.into())?;
            this.machine.env_vars.environ = Some(place);
        }

        // Collect all the pointers to each variable in a vector.
        let mut vars: Vec<Pointer<Option<Provenance>>> =
            this.machine.env_vars.map.values().copied().collect();
        // Add the trailing null pointer.
        vars.push(Pointer::null());
        // Make an array with all these pointers inside Miri.
        let tcx = this.tcx;
        let vars_layout = this.layout_of(Ty::new_array(
            tcx.tcx,
            this.machine.layouts.mut_raw_ptr.ty,
            u64::try_from(vars.len()).unwrap(),
        ))?;
        let vars_place = this.allocate(vars_layout, MiriMemoryKind::Runtime.into())?;
        for (idx, var) in vars.into_iter().enumerate() {
            let place = this.mplace_field(&vars_place, idx)?;
            this.write_pointer(var, &place.into())?;
        }
        this.write_pointer(vars_place.ptr, &this.machine.env_vars.environ.unwrap().into())?;

        Ok(())
    }

    fn getpid(&mut self) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.assert_target_os_is_unix("getpid");

        this.check_no_isolation("`getpid`")?;

        // The reason we need to do this wacky of a conversion is because
        // `libc::getpid` returns an i32, however, `std::process::id()` return an u32.
        // So we un-do the conversion that stdlib does and turn it back into an i32.
        #[allow(clippy::cast_possible_wrap)]
        Ok(std::process::id() as i32)
    }

    #[allow(non_snake_case)]
    fn GetCurrentProcessId(&mut self) -> InterpResult<'tcx, u32> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetCurrentProcessId");

        this.check_no_isolation("`GetCurrentProcessId`")?;

        Ok(std::process::id())
    }
}
