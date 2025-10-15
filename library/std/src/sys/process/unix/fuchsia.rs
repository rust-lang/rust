use libc::{c_int, size_t};

use super::common::*;
use crate::num::NonZero;
use crate::process::StdioPipes;
use crate::sys::pal::fuchsia::*;
use crate::{fmt, io, mem, ptr};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

impl Command {
    pub fn spawn(
        &mut self,
        default: Stdio,
        needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        let envp = self.capture_env();

        if self.saw_nul() {
            return Err(io::const_error!(
                io::ErrorKind::InvalidInput,
                "nul byte found in provided data",
            ));
        }

        let (ours, theirs) = self.setup_io(default, needs_stdin)?;

        let process_handle = unsafe { self.do_exec(theirs, envp.as_ref())? };

        Ok((Process { handle: Handle::new(process_handle) }, ours))
    }

    pub fn exec(&mut self, default: Stdio) -> io::Error {
        if self.saw_nul() {
            return io::const_error!(
                io::ErrorKind::InvalidInput,
                "nul byte found in provided data",
            );
        }

        match self.setup_io(default, true) {
            Ok((_, _)) => {
                // FIXME: This is tough because we don't support the exec syscalls
                unimplemented!();
            }
            Err(e) => e,
        }
    }

    unsafe fn do_exec(
        &mut self,
        stdio: ChildPipes,
        maybe_envp: Option<&CStringArray>,
    ) -> io::Result<zx_handle_t> {
        let envp = match maybe_envp {
            // None means to clone the current environment, which is done in the
            // flags below.
            None => ptr::null(),
            Some(envp) => envp.as_ptr(),
        };

        let make_action = |local_io: &ChildStdio, target_fd| -> io::Result<fdio_spawn_action_t> {
            if let Some(local_fd) = local_io.fd() {
                Ok(fdio_spawn_action_t {
                    action: FDIO_SPAWN_ACTION_TRANSFER_FD,
                    local_fd,
                    target_fd,
                    ..Default::default()
                })
            } else {
                if let ChildStdio::Null = local_io {
                    // acts as no-op
                    return Ok(Default::default());
                }

                let mut handle = ZX_HANDLE_INVALID;
                let status = fdio_fd_clone(target_fd, &mut handle);
                if status == ZX_ERR_INVALID_ARGS || status == ZX_ERR_NOT_SUPPORTED {
                    // This descriptor is closed; skip it rather than generating an
                    // error.
                    return Ok(Default::default());
                }
                zx_cvt(status)?;

                let mut cloned_fd = 0;
                zx_cvt(fdio_fd_create(handle, &mut cloned_fd))?;

                Ok(fdio_spawn_action_t {
                    action: FDIO_SPAWN_ACTION_TRANSFER_FD,
                    local_fd: cloned_fd as i32,
                    target_fd,
                    ..Default::default()
                })
            }
        };

        // Clone stdin, stdout, and stderr
        let action1 = make_action(&stdio.stdin, 0)?;
        let action2 = make_action(&stdio.stdout, 1)?;
        let action3 = make_action(&stdio.stderr, 2)?;
        let actions = [action1, action2, action3];

        // We don't want FileDesc::drop to be called on any stdio. fdio_spawn_etc
        // always consumes transferred file descriptors.
        mem::forget(stdio);

        for callback in self.get_closures().iter_mut() {
            callback()?;
        }

        let mut process_handle: zx_handle_t = 0;
        zx_cvt(fdio_spawn_etc(
            ZX_HANDLE_INVALID,
            FDIO_SPAWN_CLONE_JOB
                | FDIO_SPAWN_CLONE_LDSVC
                | FDIO_SPAWN_CLONE_NAMESPACE
                | FDIO_SPAWN_CLONE_ENVIRON // this is ignored when envp is non-null
                | FDIO_SPAWN_CLONE_UTC_CLOCK,
            self.get_program_cstr().as_ptr(),
            self.get_argv().as_ptr(),
            envp,
            actions.len() as size_t,
            actions.as_ptr(),
            &mut process_handle,
            ptr::null_mut(),
        ))?;
        // FIXME: See if we want to do something with that err_msg

        Ok(process_handle)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

pub struct Process {
    handle: Handle,
}

impl Process {
    pub fn id(&self) -> u32 {
        self.handle.raw() as u32
    }

    pub fn kill(&mut self) -> io::Result<()> {
        unsafe {
            zx_cvt(zx_task_kill(self.handle.raw()))?;
        }

        Ok(())
    }

    pub fn send_signal(&self, _signal: i32) -> io::Result<()> {
        // Fuchsia doesn't have a direct equivalent for signals
        unimplemented!()
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        let mut proc_info: zx_info_process_t = Default::default();
        let mut actual: size_t = 0;
        let mut avail: size_t = 0;

        unsafe {
            zx_cvt(zx_object_wait_one(
                self.handle.raw(),
                ZX_TASK_TERMINATED,
                ZX_TIME_INFINITE,
                ptr::null_mut(),
            ))?;
            zx_cvt(zx_object_get_info(
                self.handle.raw(),
                ZX_INFO_PROCESS,
                (&raw mut proc_info) as *mut libc::c_void,
                size_of::<zx_info_process_t>(),
                &mut actual,
                &mut avail,
            ))?;
        }
        if actual != 1 {
            return Err(io::const_error!(
                io::ErrorKind::InvalidData,
                "failed to get exit status of process",
            ));
        }
        Ok(ExitStatus(proc_info.return_code))
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        let mut proc_info: zx_info_process_t = Default::default();
        let mut actual: size_t = 0;
        let mut avail: size_t = 0;

        unsafe {
            let status =
                zx_object_wait_one(self.handle.raw(), ZX_TASK_TERMINATED, 0, ptr::null_mut());
            match status {
                0 => {} // Success
                x if x == ZX_ERR_TIMED_OUT => {
                    return Ok(None);
                }
                _ => {
                    panic!("Failed to wait on process handle: {status}");
                }
            }
            zx_cvt(zx_object_get_info(
                self.handle.raw(),
                ZX_INFO_PROCESS,
                (&raw mut proc_info) as *mut libc::c_void,
                size_of::<zx_info_process_t>(),
                &mut actual,
                &mut avail,
            ))?;
        }
        if actual != 1 {
            return Err(io::const_error!(
                io::ErrorKind::InvalidData,
                "failed to get exit status of process",
            ));
        }
        Ok(Some(ExitStatus(proc_info.return_code)))
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub struct ExitStatus(i64);

impl ExitStatus {
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        match NonZero::try_from(self.0) {
            /* was nonzero */ Ok(failure) => Err(ExitStatusError(failure)),
            /* was zero, couldn't convert */ Err(_) => Ok(()),
        }
    }

    pub fn code(&self) -> Option<i32> {
        // FIXME: support extracting return code as an i64
        self.0.try_into().ok()
    }

    pub fn signal(&self) -> Option<i32> {
        None
    }

    // FIXME: The actually-Unix implementation in unix.rs uses WSTOPSIG, WCOREDUMP et al.
    // I infer from the implementation of `success`, `code` and `signal` above that these are not
    // available on Fuchsia.
    //
    // It does not appear that Fuchsia is Unix-like enough to implement ExitStatus (or indeed many
    // other things from std::os::unix) properly. This veneer is always going to be a bodge. So
    // while I don't know if these implementations are actually correct, I think they will do for
    // now at least.
    pub fn core_dumped(&self) -> bool {
        false
    }
    pub fn stopped_signal(&self) -> Option<i32> {
        None
    }
    pub fn continued(&self) -> bool {
        false
    }

    pub fn into_raw(&self) -> c_int {
        // We don't know what someone who calls into_raw() will do with this value, but it should
        // have the conventional Unix representation. Despite the fact that this is not
        // standardised in SuS or POSIX, all Unix systems encode the signal and exit status the
        // same way. (Ie the WIFEXITED, WEXITSTATUS etc. macros have identical behavior on every
        // Unix.)
        //
        // The caller of `std::os::unix::into_raw` is probably wanting a Unix exit status, and may
        // do their own shifting and masking, or even pass the status to another computer running a
        // different Unix variant.
        //
        // The other view would be to say that the caller on Fuchsia ought to know that `into_raw`
        // will give a raw Fuchsia status (whatever that is - I don't know, personally). That is
        // not possible here because we must return a c_int because that's what Unix (including
        // SuS and POSIX) say a wait status is, but Fuchsia apparently uses a u64, so it won't
        // necessarily fit.
        //
        // It seems to me that the right answer would be to provide std::os::fuchsia with its
        // own ExitStatusExt, rather that trying to provide a not very convincing imitation of
        // Unix. Ie, std::os::unix::process:ExitStatusExt ought not to exist on Fuchsia. But
        // fixing this up that is beyond the scope of my efforts now.
        let exit_status_as_if_unix: u8 = self.0.try_into().expect("Fuchsia process return code bigger than 8 bits, but std::os::unix::ExitStatusExt::into_raw() was called to try to convert the value into a traditional Unix-style wait status, which cannot represent values greater than 255.");
        let wait_status_as_if_unix = (exit_status_as_if_unix as c_int) << 8;
        wait_status_as_if_unix
    }
}

/// Converts a raw `c_int` to a type-safe `ExitStatus` by wrapping it without copying.
impl From<c_int> for ExitStatus {
    fn from(a: c_int) -> ExitStatus {
        ExitStatus(a as i64)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exit code: {}", self.0)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatusError(NonZero<i64>);

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        ExitStatus(self.0.into())
    }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZero<i32>> {
        // fixme: affected by the same bug as ExitStatus::code()
        ExitStatus(self.0.into()).code().map(|st| st.try_into().unwrap())
    }
}
