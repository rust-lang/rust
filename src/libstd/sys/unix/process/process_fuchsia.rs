// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use libc::{self, size_t};
use mem;
use ptr;

use sys::process::zircon::{Handle, zx_handle_t};
use sys::process::process_common::*;

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

impl Command {
    pub fn spawn(&mut self, default: Stdio, needs_stdin: bool)
                 -> io::Result<(Process, StdioPipes)> {
        let envp = self.capture_env();

        if self.saw_nul() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                      "nul byte found in provided data"));
        }

        let (ours, theirs) = self.setup_io(default, needs_stdin)?;

        let process_handle = unsafe { self.do_exec(theirs, envp.as_ref())? };

        Ok((Process { handle: Handle::new(process_handle) }, ours))
    }

    pub fn exec(&mut self, default: Stdio) -> io::Error {
        if self.saw_nul() {
            return io::Error::new(io::ErrorKind::InvalidInput,
                                  "nul byte found in provided data")
        }

        match self.setup_io(default, true) {
            Ok((_, _)) => {
                // FIXME: This is tough because we don't support the exec syscalls
                unimplemented!();
            },
            Err(e) => e,
        }
    }

    unsafe fn do_exec(&mut self, stdio: ChildPipes, maybe_envp: Option<&CStringArray>)
                      -> io::Result<zx_handle_t> {
        use sys::process::zircon::*;

        let job_handle = zx_job_default();
        let envp = match maybe_envp {
            Some(envp) => envp.as_ptr(),
            None => ptr::null(),
        };

        // To make sure launchpad_destroy gets called on the launchpad if this function fails
        struct LaunchpadDestructor(*mut launchpad_t);
        impl Drop for LaunchpadDestructor {
            fn drop(&mut self) { unsafe { launchpad_destroy(self.0); } }
        }

        // Duplicate the job handle
        let mut job_copy: zx_handle_t = ZX_HANDLE_INVALID;
        zx_cvt(zx_handle_duplicate(job_handle, ZX_RIGHT_SAME_RIGHTS, &mut job_copy))?;
        // Create a launchpad
        let mut launchpad: *mut launchpad_t = ptr::null_mut();
        zx_cvt(launchpad_create(job_copy, self.get_argv()[0], &mut launchpad))?;
        let launchpad_destructor = LaunchpadDestructor(launchpad);

        // Set the process argv
        zx_cvt(launchpad_set_args(launchpad, self.get_argv().len() as i32 - 1,
                                  self.get_argv().as_ptr()))?;
        // Setup the environment vars
        zx_cvt(launchpad_set_environ(launchpad, envp))?;
        zx_cvt(launchpad_add_vdso_vmo(launchpad))?;
        // Load the executable
        zx_cvt(launchpad_elf_load(launchpad, launchpad_vmo_from_file(self.get_argv()[0])))?;
        zx_cvt(launchpad_load_vdso(launchpad, ZX_HANDLE_INVALID))?;
        zx_cvt(launchpad_clone(launchpad, LP_CLONE_FDIO_NAMESPACE | LP_CLONE_FDIO_CWD))?;

        // Clone stdin, stdout, and stderr
        if let Some(fd) = stdio.stdin.fd() {
            zx_cvt(launchpad_transfer_fd(launchpad, fd, 0))?;
        } else {
            zx_cvt(launchpad_clone_fd(launchpad, 0, 0))?;
        }
        if let Some(fd) = stdio.stdout.fd() {
            zx_cvt(launchpad_transfer_fd(launchpad, fd, 1))?;
        } else {
            zx_cvt(launchpad_clone_fd(launchpad, 1, 1))?;
        }
        if let Some(fd) = stdio.stderr.fd() {
            zx_cvt(launchpad_transfer_fd(launchpad, fd, 2))?;
        } else {
            zx_cvt(launchpad_clone_fd(launchpad, 2, 2))?;
        }

        // We don't want FileDesc::drop to be called on any stdio. It would close their fds. The
        // fds will be closed once the child process finishes.
        mem::forget(stdio);

        for callback in self.get_closures().iter_mut() {
            callback()?;
        }

        // `launchpad_go` destroys the launchpad, so we must not
        mem::forget(launchpad_destructor);

        let mut process_handle: zx_handle_t = 0;
        let mut err_msg: *const libc::c_char = ptr::null();
        zx_cvt(launchpad_go(launchpad, &mut process_handle, &mut err_msg))?;
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
        use sys::process::zircon::*;

        unsafe { zx_cvt(zx_task_kill(self.handle.raw()))?; }

        Ok(())
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        use default::Default;
        use sys::process::zircon::*;

        let mut proc_info: zx_info_process_t = Default::default();
        let mut actual: size_t = 0;
        let mut avail: size_t = 0;

        unsafe {
            zx_cvt(zx_object_wait_one(self.handle.raw(), ZX_TASK_TERMINATED,
                                      ZX_TIME_INFINITE, ptr::null_mut()))?;
            zx_cvt(zx_object_get_info(self.handle.raw(), ZX_INFO_PROCESS,
                                      &mut proc_info as *mut _ as *mut libc::c_void,
                                      mem::size_of::<zx_info_process_t>(), &mut actual,
                                      &mut avail))?;
        }
        if actual != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                                      "Failed to get exit status of process"));
        }
        Ok(ExitStatus::new(proc_info.rec.return_code))
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        use default::Default;
        use sys::process::zircon::*;

        let mut proc_info: zx_info_process_t = Default::default();
        let mut actual: size_t = 0;
        let mut avail: size_t = 0;

        unsafe {
            let status = zx_object_wait_one(self.handle.raw(), ZX_TASK_TERMINATED,
                                            0, ptr::null_mut());
            match status {
                0 => { }, // Success
                x if x == ERR_TIMED_OUT => {
                    return Ok(None);
                },
                _ => { panic!("Failed to wait on process handle: {}", status); },
            }
            zx_cvt(zx_object_get_info(self.handle.raw(), ZX_INFO_PROCESS,
                                      &mut proc_info as *mut _ as *mut libc::c_void,
                                      mem::size_of::<zx_info_process_t>(), &mut actual,
                                      &mut avail))?;
        }
        if actual != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                                      "Failed to get exit status of process"));
        }
        Ok(Some(ExitStatus::new(proc_info.rec.return_code)))
    }
}
