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
use libc;
use mem;
use ptr;

use sys::process::magenta::{Handle, launchpad_t, mx_handle_t};
use sys::process::process_common::*;

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

impl Command {
    pub fn spawn(&mut self, default: Stdio, needs_stdin: bool)
                 -> io::Result<(Process, StdioPipes)> {
        if self.saw_nul() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                      "nul byte found in provided data"));
        }

        let (ours, theirs) = self.setup_io(default, needs_stdin)?;

        let (launchpad, process_handle) = unsafe { self.do_exec(theirs)? };

        Ok((Process { launchpad: launchpad, handle: Handle::new(process_handle) }, ours))
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

    unsafe fn do_exec(&mut self, stdio: ChildPipes)
                      -> io::Result<(*mut launchpad_t, mx_handle_t)> {
        use sys::process::magenta::*;

        let job_handle = mxio_get_startup_handle(mx_hnd_info(MX_HND_TYPE_JOB, 0));
        let envp = match *self.get_envp() {
            Some(ref envp) => envp.as_ptr(),
            None => ptr::null(),
        };

        // To make sure launchpad_destroy gets called on the launchpad if this function fails
        struct LaunchpadDestructor(*mut launchpad_t);
        impl Drop for LaunchpadDestructor {
            fn drop(&mut self) { unsafe { launchpad_destroy(self.0); } }
        }

        let mut launchpad: *mut launchpad_t = ptr::null_mut();
        let launchpad_destructor = LaunchpadDestructor(launchpad);

        // Duplicate the job handle
        let mut job_copy: mx_handle_t = MX_HANDLE_INVALID;
        mx_cvt(mx_handle_duplicate(job_handle, MX_RIGHT_SAME_RIGHTS, &mut job_copy))?;
        // Create a launchpad
        mx_cvt(launchpad_create(job_copy, self.get_argv()[0], &mut launchpad))?;
        // Set the process argv
        mx_cvt(launchpad_arguments(launchpad, self.get_argv().len() as i32 - 1,
                                   self.get_argv().as_ptr()))?;
        // Setup the environment vars
        mx_cvt(launchpad_environ(launchpad, envp))?;
        mx_cvt(launchpad_add_vdso_vmo(launchpad))?;
        mx_cvt(launchpad_clone_mxio_root(launchpad))?;
        // Load the executable
        mx_cvt(launchpad_elf_load(launchpad, launchpad_vmo_from_file(self.get_argv()[0])))?;
        mx_cvt(launchpad_load_vdso(launchpad, MX_HANDLE_INVALID))?;
        mx_cvt(launchpad_clone_mxio_cwd(launchpad))?;

        // Clone stdin, stdout, and stderr
        if let Some(fd) = stdio.stdin.fd() {
            launchpad_transfer_fd(launchpad, fd, 0);
        } else {
            launchpad_clone_fd(launchpad, 0, 0);
        }
        if let Some(fd) = stdio.stdout.fd() {
            launchpad_transfer_fd(launchpad, fd, 1);
        } else {
            launchpad_clone_fd(launchpad, 1, 1);
        }
        if let Some(fd) = stdio.stderr.fd() {
            launchpad_transfer_fd(launchpad, fd, 2);
        } else {
            launchpad_clone_fd(launchpad, 2, 2);
        }

        // We don't want FileDesc::drop to be called on any stdio. It would close their fds. The
        // fds will be closed once the child process finishes.
        mem::forget(stdio);

        for callback in self.get_closures().iter_mut() {
            callback()?;
        }

        let process_handle = mx_cvt(launchpad_start(launchpad))?;

        // Successfully started the launchpad
        mem::forget(launchpad_destructor);

        Ok((launchpad, process_handle))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

pub struct Process {
    launchpad: *mut launchpad_t,
    handle: Handle,
}

impl Process {
    pub fn id(&self) -> u32 {
        self.handle.raw() as u32
    }

    pub fn kill(&mut self) -> io::Result<()> {
        use sys::process::magenta::*;

        unsafe { mx_cvt(mx_task_kill(self.handle.raw()))?; }

        Ok(())
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        use default::Default;
        use sys::process::magenta::*;

        let mut proc_info: mx_info_process_t = Default::default();
        let mut actual: mx_size_t = 0;
        let mut avail: mx_size_t = 0;

        unsafe {
            mx_cvt(mx_handle_wait_one(self.handle.raw(), MX_TASK_TERMINATED,
                                      MX_TIME_INFINITE, ptr::null_mut()))?;
            mx_cvt(mx_object_get_info(self.handle.raw(), MX_INFO_PROCESS,
                                      &mut proc_info as *mut _ as *mut libc::c_void,
                                      mem::size_of::<mx_info_process_t>(), &mut actual,
                                      &mut avail))?;
        }
        if actual != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                                      "Failed to get exit status of process"));
        }
        Ok(ExitStatus::new(proc_info.rec.return_code))
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        use sys::process::magenta::launchpad_destroy;
        unsafe { launchpad_destroy(self.launchpad); }
    }
}
