// Tests proc thread attributes by spawning a process with a custom parent process,
// then comparing the parent process ID with the expected parent process ID.

//@ run-pass
//@ only-windows
//@ needs-subprocess
//@ edition: 2021

#![feature(windows_process_extensions_raw_attribute)]

use std::os::windows::io::AsRawHandle;
use std::os::windows::process::{CommandExt, ProcThreadAttributeList};
use std::process::{Child, Command};
use std::{env, mem, ptr, thread, time};

// Make a best effort to ensure child processes always exit.
struct ProcessDropGuard(Child);
impl Drop for ProcessDropGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}

fn main() {
    if env::args().skip(1).any(|s| s == "--child") {
        child();
    } else {
        parent();
    }
}

fn parent() {
    let exe = env::current_exe().unwrap();

    let (fake_parent_id, child_parent_id) = {
        // Create a process to be our fake parent process.
        let fake_parent = Command::new(&exe).arg("--child").spawn().unwrap();
        let fake_parent = ProcessDropGuard(fake_parent);
        let parent_handle = fake_parent.0.as_raw_handle();

        // Create another process with the parent process set to the fake.
        let mut attribute_list = ProcThreadAttributeList::build()
            .attribute(PROC_THREAD_ATTRIBUTE_PARENT_PROCESS, &parent_handle)
            .finish()
            .unwrap();
        let child =
            Command::new(&exe).arg("--child").spawn_with_attributes(&mut attribute_list).unwrap();
        let child = ProcessDropGuard(child);

        // Return the fake's process id and the child's parent's id.
        (process_info(&fake_parent.0).process_id(), process_info(&child.0).parent_id())
    };

    assert_eq!(fake_parent_id, child_parent_id);
}

// A process that stays running until killed.
fn child() {
    // Don't wait forever if something goes wrong.
    thread::sleep(time::Duration::from_secs(60));
}

fn process_info(child: &Child) -> PROCESS_BASIC_INFORMATION {
    unsafe {
        let mut info: PROCESS_BASIC_INFORMATION = mem::zeroed();
        let result = NtQueryInformationProcess(
            child.as_raw_handle(),
            ProcessBasicInformation,
            ptr::from_mut(&mut info).cast(),
            mem::size_of_val(&info).try_into().unwrap(),
            ptr::null_mut(),
        );
        assert_eq!(result, 0);
        info
    }
}

// Windows API
mod winapi {
    #![allow(nonstandard_style)]
    use std::ffi::c_void;

    pub type HANDLE = *mut c_void;
    type NTSTATUS = i32;
    type PROCESSINFOCLASS = i32;

    pub const ProcessBasicInformation: i32 = 0;
    pub const PROC_THREAD_ATTRIBUTE_PARENT_PROCESS: usize = 0x00020000;
    #[repr(C)]
    pub struct PROCESS_BASIC_INFORMATION {
        pub ExitStatus: NTSTATUS,
        pub PebBaseAddress: *mut (),
        pub AffinityMask: usize,
        pub BasePriority: i32,
        pub UniqueProcessId: usize,
        pub InheritedFromUniqueProcessId: usize,
    }
    impl PROCESS_BASIC_INFORMATION {
        pub fn parent_id(&self) -> usize {
            self.InheritedFromUniqueProcessId
        }
        pub fn process_id(&self) -> usize {
            self.UniqueProcessId
        }
    }

    #[link(name = "ntdll")]
    extern "system" {
        pub fn NtQueryInformationProcess(
            ProcessHandle: HANDLE,
            ProcessInformationClass: PROCESSINFOCLASS,
            ProcessInformation: *mut c_void,
            ProcessInformationLength: u32,
            ReturnLength: *mut u32,
        ) -> NTSTATUS;
    }
}
use winapi::*;
