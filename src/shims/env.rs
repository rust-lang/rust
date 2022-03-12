use std::convert::TryFrom;
use std::env;
use std::ffi::{OsStr, OsString};
use std::io::ErrorKind;

use rustc_const_eval::interpret::Pointer;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::layout::LayoutOf;
use rustc_target::abi::Size;

use crate::*;

/// Check whether an operation that writes to a target buffer was successful.
/// Accordingly select return value.
/// Local helper function to be used in Windows shims.
fn windows_check_buffer_size((success, len): (bool, u64)) -> u32 {
    if success {
        // If the function succeeds, the return value is the number of characters stored in the target buffer,
        // not including the terminating null character.
        u32::try_from(len).unwrap()
    } else {
        // If the target buffer was not large enough to hold the data, the return value is the buffer size, in characters,
        // required to hold the string and its terminating null character.
        u32::try_from(len.checked_add(1).unwrap()).unwrap()
    }
}

#[derive(Default)]
pub struct EnvVars<'tcx> {
    /// Stores pointers to the environment variables. These variables must be stored as
    /// null-terminated target strings (c_str or wide_str) with the `"{name}={value}"` format.
    map: FxHashMap<OsString, Pointer<Option<Tag>>>,

    /// Place where the `environ` static is stored. Lazily initialized, but then never changes.
    pub(crate) environ: Option<MPlaceTy<'tcx, Tag>>,
}

impl<'tcx> EnvVars<'tcx> {
    pub(crate) fn init<'mir>(
        ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'mir, 'tcx>>,
        mut excluded_env_vars: Vec<String>,
        forwarded_env_vars: Vec<String>,
    ) -> InterpResult<'tcx> {
        let target_os = ecx.tcx.sess.target.os.as_str();
        // HACK: Exclude `TERM` var to avoid terminfo trying to open the termcap file.
        // This is (a) very slow and (b) does not work on Windows.
        excluded_env_vars.push("TERM".to_owned());

        // Skip the loop entirely if we don't want to forward anything.
        if ecx.machine.communicate() || !forwarded_env_vars.is_empty() {
            for (name, value) in env::vars_os() {
                let forward = match ecx.machine.communicate() {
                    true => !excluded_env_vars.iter().any(|v| v.as_str() == &name),
                    false => forwarded_env_vars.iter().any(|v| v.as_str() == &name),
                };
                if forward {
                    let var_ptr = match target_os {
                        "linux" | "macos" =>
                            alloc_env_var_as_c_str(name.as_ref(), value.as_ref(), ecx)?,
                        "windows" => alloc_env_var_as_wide_str(name.as_ref(), value.as_ref(), ecx)?,
                        unsupported =>
                            throw_unsup_format!(
                                "environment support for target OS `{}` not yet available",
                                unsupported
                            ),
                    };
                    ecx.machine.env_vars.map.insert(OsString::from(name), var_ptr);
                }
            }
        }
        ecx.update_environ()
    }

    pub(crate) fn cleanup<'mir>(
        ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'mir, 'tcx>>,
    ) -> InterpResult<'tcx> {
        // Deallocate individual env vars.
        for (_name, ptr) in ecx.machine.env_vars.map.drain() {
            ecx.memory.deallocate(ptr, None, MiriMemoryKind::Runtime.into())?;
        }
        // Deallocate environ var list.
        let environ = ecx.machine.env_vars.environ.unwrap();
        let old_vars_ptr = ecx.read_pointer(&environ.into())?;
        ecx.memory.deallocate(old_vars_ptr, None, MiriMemoryKind::Runtime.into())?;
        Ok(())
    }
}

fn alloc_env_var_as_c_str<'mir, 'tcx>(
    name: &OsStr,
    value: &OsStr,
    ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'mir, 'tcx>>,
) -> InterpResult<'tcx, Pointer<Option<Tag>>> {
    let mut name_osstring = name.to_os_string();
    name_osstring.push("=");
    name_osstring.push(value);
    ecx.alloc_os_str_as_c_str(name_osstring.as_os_str(), MiriMemoryKind::Runtime.into())
}

fn alloc_env_var_as_wide_str<'mir, 'tcx>(
    name: &OsStr,
    value: &OsStr,
    ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'mir, 'tcx>>,
) -> InterpResult<'tcx, Pointer<Option<Tag>>> {
    let mut name_osstring = name.to_os_string();
    name_osstring.push("=");
    name_osstring.push(value);
    ecx.alloc_os_str_as_wide_str(name_osstring.as_os_str(), MiriMemoryKind::Runtime.into())
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn getenv(&mut self, name_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, Pointer<Option<Tag>>> {
        let this = self.eval_context_mut();
        let target_os = &this.tcx.sess.target.os;
        assert!(
            target_os == "linux" || target_os == "macos",
            "`getenv` is only available for the UNIX target family"
        );

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
        name_op: &OpTy<'tcx, Tag>, // LPCWSTR
        buf_op: &OpTy<'tcx, Tag>,  // LPWSTR
        size_op: &OpTy<'tcx, Tag>, // DWORD
    ) -> InterpResult<'tcx, u32> {
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
                windows_check_buffer_size(this.write_os_str_to_wide_str(&var, buf_ptr, buf_size)?)
            }
            None => {
                let envvar_not_found = this.eval_windows("c", "ERROR_ENVVAR_NOT_FOUND")?;
                this.set_last_error(envvar_not_found)?;
                0 // return zero upon failure
            }
        })
    }

    #[allow(non_snake_case)]
    fn GetEnvironmentStringsW(&mut self) -> InterpResult<'tcx, Pointer<Option<Tag>>> {
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
        env_block_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "FreeEnvironmentStringsW");

        let env_block_ptr = this.read_pointer(env_block_op)?;
        let result = this.memory.deallocate(env_block_ptr, None, MiriMemoryKind::Runtime.into());
        // If the function succeeds, the return value is nonzero.
        Ok(result.is_ok() as i32)
    }

    fn setenv(
        &mut self,
        name_op: &OpTy<'tcx, Tag>,
        value_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let mut this = self.eval_context_mut();
        let target_os = &this.tcx.sess.target.os;
        assert!(
            target_os == "linux" || target_os == "macos",
            "`setenv` is only available for the UNIX target family"
        );

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
            let var_ptr = alloc_env_var_as_c_str(&name, &value, &mut this)?;
            if let Some(var) = this.machine.env_vars.map.insert(name, var_ptr) {
                this.memory.deallocate(var, None, MiriMemoryKind::Runtime.into())?;
            }
            this.update_environ()?;
            Ok(0) // return zero on success
        } else {
            // name argument is a null pointer, points to an empty string, or points to a string containing an '=' character.
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            Ok(-1)
        }
    }

    #[allow(non_snake_case)]
    fn SetEnvironmentVariableW(
        &mut self,
        name_op: &OpTy<'tcx, Tag>,  // LPCWSTR
        value_op: &OpTy<'tcx, Tag>, // LPCWSTR
    ) -> InterpResult<'tcx, i32> {
        let mut this = self.eval_context_mut();
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
                this.memory.deallocate(var, None, MiriMemoryKind::Runtime.into())?;
                this.update_environ()?;
            }
            Ok(1) // return non-zero on success
        } else {
            let value = this.read_os_str_from_wide_str(value_ptr)?;
            let var_ptr = alloc_env_var_as_wide_str(&name, &value, &mut this)?;
            if let Some(var) = this.machine.env_vars.map.insert(name, var_ptr) {
                this.memory.deallocate(var, None, MiriMemoryKind::Runtime.into())?;
            }
            this.update_environ()?;
            Ok(1) // return non-zero on success
        }
    }

    fn unsetenv(&mut self, name_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        let target_os = &this.tcx.sess.target.os;
        assert!(
            target_os == "linux" || target_os == "macos",
            "`unsetenv` is only available for the UNIX target family"
        );

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
                this.memory.deallocate(var, None, MiriMemoryKind::Runtime.into())?;
            }
            this.update_environ()?;
            Ok(0)
        } else {
            // name argument is a null pointer, points to an empty string, or points to a string containing an '=' character.
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            Ok(-1)
        }
    }

    fn getcwd(
        &mut self,
        buf_op: &OpTy<'tcx, Tag>,
        size_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, Pointer<Option<Tag>>> {
        let this = self.eval_context_mut();
        let target_os = &this.tcx.sess.target.os;
        assert!(
            target_os == "linux" || target_os == "macos",
            "`getcwd` is only available for the UNIX target family"
        );

        let buf = this.read_pointer(&buf_op)?;
        let size = this.read_scalar(&size_op)?.to_machine_usize(&*this.tcx)?;

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
                let erange = this.eval_libc("ERANGE")?;
                this.set_last_error(erange)?;
            }
            Err(e) => this.set_last_error_from_io_error(e.kind())?,
        }

        Ok(Pointer::null())
    }

    #[allow(non_snake_case)]
    fn GetCurrentDirectoryW(
        &mut self,
        size_op: &OpTy<'tcx, Tag>, // DWORD
        buf_op: &OpTy<'tcx, Tag>,  // LPTSTR
    ) -> InterpResult<'tcx, u32> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetCurrentDirectoryW");

        let size = u64::from(this.read_scalar(size_op)?.to_u32()?);
        let buf = this.read_pointer(buf_op)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`GetCurrentDirectoryW`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;
            return Ok(0);
        }

        // If we cannot get the current directory, we return 0
        match env::current_dir() {
            Ok(cwd) =>
                return Ok(windows_check_buffer_size(this.write_path_to_wide_str(&cwd, buf, size)?)),
            Err(e) => this.set_last_error_from_io_error(e.kind())?,
        }
        Ok(0)
    }

    fn chdir(&mut self, path_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        let target_os = &this.tcx.sess.target.os;
        assert!(
            target_os == "linux" || target_os == "macos",
            "`getcwd` is only available for the UNIX target family"
        );

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
        path_op: &OpTy<'tcx, Tag>, // LPCTSTR
    ) -> InterpResult<'tcx, i32> {
        // ^ Returns BOOL (i32 on Windows)

        let this = self.eval_context_mut();
        this.assert_target_os("windows", "SetCurrentDirectoryW");

        let path = this.read_path_from_wide_str(this.read_pointer(path_op)?)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`SetCurrentDirectoryW`", reject_with)?;
            this.set_last_error_from_io_error(ErrorKind::PermissionDenied)?;

            return Ok(0);
        }

        match env::set_current_dir(path) {
            Ok(()) => Ok(1),
            Err(e) => {
                this.set_last_error_from_io_error(e.kind())?;
                Ok(0)
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
            this.memory.deallocate(old_vars_ptr, None, MiriMemoryKind::Runtime.into())?;
        } else {
            // No `environ` allocated yet, let's do that.
            // This is memory backing an extern static, hence `ExternStatic`, not `Env`.
            let layout = this.machine.layouts.usize;
            let place = this.allocate(layout, MiriMemoryKind::ExternStatic.into())?;
            this.machine.env_vars.environ = Some(place);
        }

        // Collect all the pointers to each variable in a vector.
        let mut vars: Vec<Pointer<Option<Tag>>> =
            this.machine.env_vars.map.values().copied().collect();
        // Add the trailing null pointer.
        vars.push(Pointer::null());
        // Make an array with all these pointers inside Miri.
        let tcx = this.tcx;
        let vars_layout =
            this.layout_of(tcx.mk_array(tcx.types.usize, u64::try_from(vars.len()).unwrap()))?;
        let vars_place = this.allocate(vars_layout, MiriMemoryKind::Runtime.into())?;
        for (idx, var) in vars.into_iter().enumerate() {
            let place = this.mplace_field(&vars_place, idx)?;
            this.write_pointer(var, &place.into())?;
        }
        this.write_pointer(vars_place.ptr, &this.machine.env_vars.environ.unwrap().into())?;

        Ok(())
    }
}
