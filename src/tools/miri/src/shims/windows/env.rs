use std::env;
use std::ffi::{OsStr, OsString};
use std::io::ErrorKind;

use rustc_data_structures::fx::FxHashMap;

use self::helpers::windows_check_buffer_size;
use crate::*;

#[derive(Default)]
pub struct WindowsEnvVars {
    /// Stores the environment variables.
    map: FxHashMap<OsString, OsString>,
}

impl VisitProvenance for WindowsEnvVars {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        let WindowsEnvVars { map: _ } = self;
    }
}

impl WindowsEnvVars {
    pub(crate) fn new<'tcx>(
        _ecx: &mut InterpCx<'tcx, MiriMachine<'tcx>>,
        env_vars: FxHashMap<OsString, OsString>,
    ) -> InterpResult<'tcx, Self> {
        interp_ok(Self { map: env_vars })
    }

    /// Implementation detail for [`InterpCx::get_env_var`].
    pub(crate) fn get<'tcx>(&self, name: &OsStr) -> InterpResult<'tcx, Option<OsString>> {
        interp_ok(self.map.get(name).cloned())
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    #[allow(non_snake_case)]
    fn GetEnvironmentVariableW(
        &mut self,
        name_op: &OpTy<'tcx>, // LPCWSTR
        buf_op: &OpTy<'tcx>,  // LPWSTR
        size_op: &OpTy<'tcx>, // DWORD
    ) -> InterpResult<'tcx, Scalar> {
        // ^ Returns DWORD (u32 on Windows)

        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetEnvironmentVariableW");

        let name_ptr = this.read_pointer(name_op)?;
        let buf_ptr = this.read_pointer(buf_op)?;
        let buf_size = this.read_scalar(size_op)?.to_u32()?; // in characters

        let name = this.read_os_str_from_wide_str(name_ptr)?;
        interp_ok(match this.machine.env_vars.windows().map.get(&name).cloned() {
            Some(val) => {
                Scalar::from_u32(windows_check_buffer_size(this.write_os_str_to_wide_str(
                    &val,
                    buf_ptr,
                    buf_size.into(),
                )?))
                // This can in fact return 0. It is up to the caller to set last_error to 0
                // beforehand and check it afterwards to exclude that case.
            }
            None => {
                let envvar_not_found = this.eval_windows("c", "ERROR_ENVVAR_NOT_FOUND");
                this.set_last_error(envvar_not_found)?;
                Scalar::from_u32(0) // return zero upon failure
            }
        })
    }

    #[allow(non_snake_case)]
    fn GetEnvironmentStringsW(&mut self) -> InterpResult<'tcx, Pointer> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetEnvironmentStringsW");

        // Info on layout of environment blocks in Windows:
        // https://docs.microsoft.com/en-us/windows/win32/procthread/environment-variables
        let mut env_vars = std::ffi::OsString::new();
        for (name, value) in this.machine.env_vars.windows().map.iter() {
            env_vars.push(name);
            env_vars.push("=");
            env_vars.push(value);
            env_vars.push("\0");
        }
        // Allocate environment block & Store environment variables to environment block.
        // Final null terminator(block terminator) is added by `alloc_os_str_to_wide_str`.
        let envblock_ptr =
            this.alloc_os_str_as_wide_str(&env_vars, MiriMemoryKind::Runtime.into())?;
        // If the function succeeds, the return value is a pointer to the environment block of the current process.
        interp_ok(envblock_ptr)
    }

    #[allow(non_snake_case)]
    fn FreeEnvironmentStringsW(&mut self, env_block_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "FreeEnvironmentStringsW");

        let env_block_ptr = this.read_pointer(env_block_op)?;
        this.deallocate_ptr(env_block_ptr, None, MiriMemoryKind::Runtime.into())?;
        // If the function succeeds, the return value is nonzero.
        interp_ok(Scalar::from_i32(1))
    }

    #[allow(non_snake_case)]
    fn SetEnvironmentVariableW(
        &mut self,
        name_op: &OpTy<'tcx>,  // LPCWSTR
        value_op: &OpTy<'tcx>, // LPCWSTR
    ) -> InterpResult<'tcx, Scalar> {
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
            // Delete environment variable `{name}` if it exists.
            this.machine.env_vars.windows_mut().map.remove(&name);
            interp_ok(this.eval_windows("c", "TRUE"))
        } else {
            let value = this.read_os_str_from_wide_str(value_ptr)?;
            this.machine.env_vars.windows_mut().map.insert(name, value);
            interp_ok(this.eval_windows("c", "TRUE"))
        }
    }

    #[allow(non_snake_case)]
    fn GetCurrentDirectoryW(
        &mut self,
        size_op: &OpTy<'tcx>, // DWORD
        buf_op: &OpTy<'tcx>,  // LPTSTR
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetCurrentDirectoryW");

        let size = u64::from(this.read_scalar(size_op)?.to_u32()?);
        let buf = this.read_pointer(buf_op)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`GetCurrentDirectoryW`", reject_with)?;
            this.set_last_error(ErrorKind::PermissionDenied)?;
            return interp_ok(Scalar::from_u32(0));
        }

        // If we cannot get the current directory, we return 0
        match env::current_dir() {
            Ok(cwd) => {
                // This can in fact return 0. It is up to the caller to set last_error to 0
                // beforehand and check it afterwards to exclude that case.
                return interp_ok(Scalar::from_u32(windows_check_buffer_size(
                    this.write_path_to_wide_str(&cwd, buf, size)?,
                )));
            }
            Err(e) => this.set_last_error(e)?,
        }
        interp_ok(Scalar::from_u32(0))
    }

    #[allow(non_snake_case)]
    fn SetCurrentDirectoryW(
        &mut self,
        path_op: &OpTy<'tcx>, // LPCTSTR
    ) -> InterpResult<'tcx, Scalar> {
        // ^ Returns BOOL (i32 on Windows)

        let this = self.eval_context_mut();
        this.assert_target_os("windows", "SetCurrentDirectoryW");

        let path = this.read_path_from_wide_str(this.read_pointer(path_op)?)?;

        if let IsolatedOp::Reject(reject_with) = this.machine.isolated_op {
            this.reject_in_isolation("`SetCurrentDirectoryW`", reject_with)?;
            this.set_last_error(ErrorKind::PermissionDenied)?;

            return interp_ok(this.eval_windows("c", "FALSE"));
        }

        match env::set_current_dir(path) {
            Ok(()) => interp_ok(this.eval_windows("c", "TRUE")),
            Err(e) => {
                this.set_last_error(e)?;
                interp_ok(this.eval_windows("c", "FALSE"))
            }
        }
    }

    #[allow(non_snake_case)]
    fn GetCurrentProcessId(&mut self) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetCurrentProcessId");

        interp_ok(Scalar::from_u32(this.get_pid()))
    }

    #[allow(non_snake_case)]
    fn GetUserProfileDirectoryW(
        &mut self,
        token: &OpTy<'tcx>, // HANDLE
        buf: &OpTy<'tcx>,   // LPWSTR
        size: &OpTy<'tcx>,  // LPDWORD
    ) -> InterpResult<'tcx, Scalar> // returns BOOL
    {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetUserProfileDirectoryW");
        this.check_no_isolation("`GetUserProfileDirectoryW`")?;

        let token = this.read_target_isize(token)?;
        let buf = this.read_pointer(buf)?;
        let size = this.deref_pointer_as(size, this.machine.layouts.u32)?;

        if token != -4 {
            throw_unsup_format!(
                "GetUserProfileDirectoryW: only CURRENT_PROCESS_TOKEN is supported"
            );
        }

        // See <https://learn.microsoft.com/en-us/windows/win32/api/userenv/nf-userenv-getuserprofiledirectoryw> for docs.
        interp_ok(match directories::UserDirs::new() {
            Some(dirs) => {
                let home = dirs.home_dir();
                let size_avail = if this.ptr_is_null(buf)? {
                    0 // if the buf pointer is null, we can't write to it; `size` will be updated to the required length
                } else {
                    this.read_scalar(&size)?.to_u32()?
                };
                // Of course we cannot use `windows_check_buffer_size` here since this uses
                // a different method for dealing with a too-small buffer than the other functions...
                let (success, len) = this.write_path_to_wide_str(home, buf, size_avail.into())?;
                // As per <https://github.com/MicrosoftDocs/sdk-api/pull/1810>, the size is always
                // written, not just on failure.
                this.write_scalar(Scalar::from_u32(len.try_into().unwrap()), &size)?;
                if success {
                    Scalar::from_i32(1) // return TRUE
                } else {
                    this.set_last_error(this.eval_windows("c", "ERROR_INSUFFICIENT_BUFFER"))?;
                    Scalar::from_i32(0) // return FALSE
                }
            }
            None => {
                // We have to pick some error code.
                this.set_last_error(this.eval_windows("c", "ERROR_BAD_USER_PROFILE"))?;
                Scalar::from_i32(0) // return FALSE
            }
        })
    }
}
