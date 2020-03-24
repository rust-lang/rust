use std::ffi::{OsString, OsStr};
use std::env;
use std::convert::TryFrom;
use std::collections::hash_map::Values;

use crate::stacked_borrows::Tag;
use crate::rustc_target::abi::LayoutOf;
use crate::*;

use rustc_data_structures::fx::FxHashMap;
use rustc::ty::layout::Size;
use rustc_mir::interpret::Pointer;

#[derive(Default)]
pub struct EnvVars<'tcx> {
    /// Stores pointers to the environment variables. These variables must be stored as
    /// null-terminated target strings (c_str or wide_str) with the `"{name}={value}"` format.
    map: FxHashMap<OsString, Pointer<Tag>>,

    /// Place where the `environ` static is stored. Lazily initialized, but then never changes.
    pub(crate) environ: Option<MPlaceTy<'tcx, Tag>>,
}

impl<'tcx> EnvVars<'tcx> {
    pub(crate) fn init<'mir>(
        ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
        mut excluded_env_vars: Vec<String>,
    ) -> InterpResult<'tcx> {
        if ecx.tcx.sess.target.target.target_os.to_lowercase() == "windows" {
            // Exclude `TERM` var to avoid terminfo trying to open the termcap file.
            excluded_env_vars.push("TERM".to_owned());
        }
        if ecx.machine.communicate {
            let target_os = ecx.tcx.sess.target.target.target_os.as_str();
            for (name, value) in env::vars() {
                if !excluded_env_vars.contains(&name) {
                    let var_ptr = match target_os {
                        "linux" | "macos" => alloc_env_var_as_c_str(name.as_ref(), value.as_ref(), ecx)?,
                        "windows" => alloc_env_var_as_wide_str(name.as_ref(), value.as_ref(), ecx)?,
                        unsupported => throw_unsup_format!("environment support for target OS `{}` not yet available", unsupported),
                    };
                    ecx.machine.env_vars.map.insert(OsString::from(name), var_ptr);
                }
            }
        }
        ecx.update_environ()
    }

    pub(super) fn values(&self) -> InterpResult<'tcx, Values<'_, OsString, Pointer<Tag>>> {
        Ok(self.map.values())
    }
}

fn alloc_env_var_as_c_str<'mir, 'tcx>(
    name: &OsStr,
    value: &OsStr,
    ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
) -> InterpResult<'tcx, Pointer<Tag>> {
    let mut name_osstring = name.to_os_string();
    name_osstring.push("=");
    name_osstring.push(value);
    Ok(ecx.alloc_os_str_as_c_str(name_osstring.as_os_str(), MiriMemoryKind::Machine.into()))
}

fn alloc_env_var_as_wide_str<'mir, 'tcx>(
    name: &OsStr,
    value: &OsStr,
    ecx: &mut InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
) -> InterpResult<'tcx, Pointer<Tag>> {
    let mut name_osstring = name.to_os_string();
    name_osstring.push("=");
    name_osstring.push(value);
    Ok(ecx.alloc_os_str_as_wide_str(name_osstring.as_os_str(), MiriMemoryKind::Machine.into()))
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn getenv(&mut self, name_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();
        let target_os = &this.tcx.sess.target.target.target_os;
        assert!(target_os == "linux" || target_os == "macos", "`getenv` is only available for the UNIX target family");

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let name = this.read_os_str_from_c_str(name_ptr)?;
        Ok(match this.machine.env_vars.map.get(name) {
            Some(var_ptr) => {
                // The offset is used to strip the "{name}=" part of the string.
                Scalar::from(var_ptr.offset(Size::from_bytes(u64::try_from(name.len()).unwrap().checked_add(1).unwrap()), this)?)
            }
            None => Scalar::ptr_null(&*this.tcx),
        })
    }

    fn getenvironmentvariablew(
        &mut self,
        name_op: OpTy<'tcx, Tag>, // LPCWSTR lpName
        buf_op: OpTy<'tcx, Tag>, // LPWSTR  lpBuffer
        size_op: OpTy<'tcx, Tag>, // DWORD   nSize
    ) -> InterpResult<'tcx, u64> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetEnvironmentVariableW");

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let name = this.read_os_str_from_wide_str(name_ptr)?;
        Ok(match this.machine.env_vars.map.get(&name) {
            Some(var_ptr) => {
                // The offset is used to strip the "{name}=" part of the string.
                let name_offset_bytes =
                    u64::try_from(name.len()).unwrap().checked_add(1).unwrap().checked_mul(2).unwrap();
                let var_ptr = Scalar::from(var_ptr.offset(Size::from_bytes(name_offset_bytes), this)?);

                let var_size = u64::try_from(this.read_os_str_from_wide_str(var_ptr)?.len()).unwrap();
                let buf_size = u64::try_from(this.read_scalar(size_op)?.to_i32()?).unwrap();
                let return_val = if var_size.checked_add(1).unwrap() > buf_size {
                    // If lpBuffer is not large enough to hold the data, the return value is the buffer size, in characters,
                    // required to hold the string and its terminating null character and the contents of lpBuffer are undefined.
                    var_size + 1
                } else {
                    let buf_ptr = this.read_scalar(buf_op)?.not_undef()?;
                    for i in 0..var_size {
                        this.memory.copy(
                            this.force_ptr(var_ptr.ptr_offset(Size::from_bytes(i) * 2, this)?)?,
                            this.force_ptr(buf_ptr.ptr_offset(Size::from_bytes(i) * 2, this)?)?,
                            Size::from_bytes(2),
                            true,
                        )?;
                    }
                    // If the function succeeds, the return value is the number of characters stored in the buffer pointed to by lpBuffer,
                    // not including the terminating null character.
                    var_size
                };
                return_val
            }
            None => {
                this.set_last_error(Scalar::from_u32(203))?; // ERROR_ENVVAR_NOT_FOUND
                0 // return zero upon failure
            }
        })
    }

    fn getenvironmentstringsw(&mut self) -> InterpResult<'tcx, Scalar<Tag>> {
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetEnvironmentStringsW");

        // Info on layout of environment blocks in Windows: 
        // https://docs.microsoft.com/en-us/windows/win32/procthread/environment-variables
        let mut env_vars = std::ffi::OsString::new();
        for &item in this.machine.env_vars.values()? {
            let env_var = this.read_os_str_from_target_str(Scalar::from(item))?;
            env_vars.push(env_var);
            env_vars.push("\0");
        }

        // Allocate environment block
        let tcx = this.tcx;
        let env_block_size = env_vars.len().checked_add(1).unwrap();
        let env_block_type = tcx.mk_array(tcx.types.u16, u64::try_from(env_block_size).unwrap());
        let env_block_place = this.allocate(this.layout_of(env_block_type)?, MiriMemoryKind::WinHeap.into());
        
        // Store environment variables to environment block
        // Final null terminator(block terminator) is pushed by `write_os_str_to_wide_str`
        this.write_os_str_to_wide_str(&env_vars, env_block_place, u64::try_from(env_block_size).unwrap())?;

        Ok(env_block_place.ptr)
    }

    fn setenv(
        &mut self,
        name_op: OpTy<'tcx, Tag>,
        value_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let mut this = self.eval_context_mut();
        let target_os = &this.tcx.sess.target.target.target_os;
        assert!(target_os == "linux" || target_os == "macos", "`setenv` is only available for the UNIX target family");

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let value_ptr = this.read_scalar(value_op)?.not_undef()?;

        let mut new = None;
        if !this.is_null(name_ptr)? {
            let name = this.read_os_str_from_c_str(name_ptr)?;
            if !name.is_empty() && !name.to_string_lossy().contains('=') {
                let value = this.read_os_str_from_c_str(value_ptr)?;
                new = Some((name.to_owned(), value.to_owned()));
            }
        }
        if let Some((name, value)) = new {
            let var_ptr = alloc_env_var_as_c_str(&name, &value, &mut this)?;
            if let Some(var) = this.machine.env_vars.map.insert(name, var_ptr) {
                this.memory
                    .deallocate(var, None, MiriMemoryKind::Machine.into())?;
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

    fn setenvironmentvariablew(
        &mut self,
        name_op: OpTy<'tcx, Tag>, // LPCWSTR lpName,
        value_op: OpTy<'tcx, Tag>, // LPCWSTR lpValue,
    ) -> InterpResult<'tcx, i32> {
        let mut this = self.eval_context_mut();
        this.assert_target_os("windows", "SetEnvironmentVariableW");

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let value_ptr = this.read_scalar(value_op)?.not_undef()?;

        let mut new = None;
        if !this.is_null(name_ptr)? {
            let name = this.read_os_str_from_target_str(name_ptr)?;
            if this.is_null(value_ptr)? {
                // Delete environment variable `{name}`
                if let Some(var) = this.machine.env_vars.map.remove(&name) {
                    this.memory.deallocate(var, None, MiriMemoryKind::Machine.into())?;
                    this.update_environ()?;
                }
                return Ok(1);  // return non-zero on success
            }
            if !name.is_empty() && !name.to_string_lossy().contains('=') {
                let value = this.read_os_str_from_target_str(value_ptr)?;
                new = Some((name.to_owned(), value.to_owned()));
            }
        }
        if let Some((name, value)) = new {
            let var_ptr = alloc_env_var_as_target_str(&name, &value, &mut this)?;
            if let Some(var) = this.machine.env_vars.map.insert(name, var_ptr) {
                this.memory
                    .deallocate(var, None, MiriMemoryKind::Machine.into())?;
            }
            this.update_environ()?;
            Ok(1) // return non-zero on success
        } else {
            Ok(0)
        }
    }

    fn unsetenv(&mut self, name_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        let target_os = &this.tcx.sess.target.target.target_os;
        assert!(target_os == "linux" || target_os == "macos", "`unsetenv` is only available for the UNIX target family");

        let name_ptr = this.read_scalar(name_op)?.not_undef()?;
        let mut success = None;
        if !this.is_null(name_ptr)? {
            let name = this.read_os_str_from_target_str(name_ptr)?.to_owned();
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
            // name argument is a null pointer, points to an empty string, or points to a string containing an '=' character.
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
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
                if this.write_path_to_c_str(&cwd, buf, size)?.0 {
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

        let path = this.read_path_from_c_str(this.read_scalar(path_op)?.not_undef()?)?;

        match env::set_current_dir(path) {
            Ok(()) => Ok(0),
            Err(e) => {
                this.set_last_error_from_io_error(e)?;
                Ok(-1)
            }
        }
    }

    /// Updates the `environ` static.
    /// The first time it gets called, also initializes `extra.environ`.
    fn update_environ(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // Deallocate the old environ value, if any.
        if let Some(environ) = this.machine.env_vars.environ {
            let old_vars_ptr = this.read_scalar(environ.into())?.not_undef()?;
            this.memory.deallocate(this.force_ptr(old_vars_ptr)?, None, MiriMemoryKind::Machine.into())?;
        } else {
            // No `environ` allocated yet, let's do that.
            let layout = this.layout_of(this.tcx.types.usize)?;
            let place = this.allocate(layout, MiriMemoryKind::Machine.into());
            this.write_scalar(Scalar::from_machine_usize(0, &*this.tcx), place.into())?;
            this.machine.env_vars.environ = Some(place);
        }

        // Collect all the pointers to each variable in a vector.
        let mut vars: Vec<Scalar<Tag>> = this.machine.env_vars.map.values().map(|&ptr| ptr.into()).collect();
        // Add the trailing null pointer.
        vars.push(Scalar::from_int(0, this.pointer_size()));
        // Make an array with all these pointers inside Miri.
        let tcx = this.tcx;
        let vars_layout =
            this.layout_of(tcx.mk_array(tcx.types.usize, u64::try_from(vars.len()).unwrap()))?;
        let vars_place = this.allocate(vars_layout, MiriMemoryKind::Machine.into());
        for (idx, var) in vars.into_iter().enumerate() {
            let place = this.mplace_field(vars_place, idx)?;
            this.write_scalar(var, place.into())?;
        }
        this.write_scalar(
            vars_place.ptr,
            this.machine.env_vars.environ.unwrap().into(),
        )?;

        Ok(())
    }
}
