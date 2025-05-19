#[cfg(windows)]
pub use for_windows::*;

#[cfg(any(target_os = "haiku", target_os = "hermit", not(any(unix, windows))))]
pub unsafe fn setup(_build: &mut crate::Build) {}

#[cfg(all(unix, not(target_os = "haiku")))]
pub unsafe fn setup(build: &mut crate::Build) {
    if build.config.low_priority {
        unsafe {
            libc::setpriority(libc::PRIO_PGRP as _, 0, 10);
        }
    }
}

/// Job management on Windows for bootstrapping
///
/// Most of the time when you're running a build system (e.g., make) you expect
/// Ctrl-C or abnormal termination to actually terminate the entire tree of
/// processes in play. This currently works "by
/// default" on Unix platforms because Ctrl-C actually sends a signal to the
/// *process group* so everything will get torn
/// down. On Windows, however, Ctrl-C is only sent to processes in the same console.
/// If a process is detached or attached to another console, it won't receive the
/// signal.
///
/// To achieve the same semantics on Windows we use Job Objects to ensure that
/// all processes die at the same time. Job objects have a mode of operation
/// where when all handles to the object are closed it causes all child
/// processes associated with the object to be terminated immediately.
/// Conveniently whenever a process in the job object spawns a new process the
/// child will be associated with the job object as well. This means if we add
/// ourselves to the job object we create then everything will get torn down!
///
/// Unfortunately most of the time the build system is actually called from a
/// python wrapper (which manages things like building the build system) so this
/// all doesn't quite cut it so far. To go the last mile we duplicate the job
/// object handle into our parent process (a python process probably) and then
/// close our own handle. This means that the only handle to the job object
/// resides in the parent python process, so when python dies the whole build
/// system dies (as one would probably expect!).
///
/// Note that this is a Windows specific module as none of this logic is required on Unix.
#[cfg(windows)]
mod for_windows {
    use std::ffi::c_void;
    use std::io;

    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Diagnostics::Debug::{
        SEM_NOGPFAULTERRORBOX, SetErrorMode, THREAD_ERROR_MODE,
    };
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOB_OBJECT_LIMIT_PRIORITY_CLASS, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectExtendedLimitInformation, SetInformationJobObject,
    };
    use windows::Win32::System::Threading::{BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess};
    use windows::core::PCWSTR;

    use crate::Build;

    pub unsafe fn setup(build: &mut Build) {
        // SAFETY: pretty much everything below is unsafe
        unsafe {
            // Enable the Windows Error Reporting dialog which msys disables,
            // so we can JIT debug rustc
            let mode = SetErrorMode(THREAD_ERROR_MODE::default());
            SetErrorMode(mode & !SEM_NOGPFAULTERRORBOX);

            // Create a new job object for us to use
            let job = CreateJobObjectW(None, PCWSTR::null()).unwrap();

            // Indicate that when all handles to the job object are gone that all
            // process in the object should be killed. Note that this includes our
            // entire process tree by default because we've added ourselves and our
            // children will reside in the job by default.
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if build.config.low_priority {
                info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PRIORITY_CLASS;
                info.BasicLimitInformation.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS.0;
            }
            let r = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const c_void,
                size_of_val(&info) as u32,
            );
            assert!(r.is_ok(), "{}", io::Error::last_os_error());

            // Assign our process to this job object.
            let r = AssignProcessToJobObject(job, GetCurrentProcess());
            if r.is_err() {
                CloseHandle(job).ok();
                return;
            }
        }

        // Note: we intentionally leak the job object handle. When our process exits
        // (normally or abnormally) it will close the handle implicitly, causing all
        // processes in the job to be cleaned up.
    }
}
