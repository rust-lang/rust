// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use core::prelude::*;
use libc;
use local::Local;
use task::Task;

pub unsafe fn init() {
    imp::init();
}

pub unsafe fn cleanup() {
    imp::cleanup();
}

pub struct Handler {
    _data: *mut libc::c_void
}

impl Handler {
    pub unsafe fn new() -> Handler {
        imp::make_handler()
    }
}

impl Drop for Handler {
    fn drop(&mut self) {
        unsafe {
            imp::drop_handler(self);
        }
    }
}

pub unsafe fn report() {
    // See the message below for why this is not emitted to the
    // ^ Where did the message below go?
    // task's logger. This has the additional conundrum of the
    // logger may not be initialized just yet, meaning that an FFI
    // call would happen to initialized it (calling out to libuv),
    // and the FFI call needs 2MB of stack when we just ran out.

    let task: Option<*mut Task> = Local::try_unsafe_borrow();

    let name = task.and_then(|task| {
        (*task).name.as_ref().map(|n| n.as_slice())
    });

    rterrln!("\ntask '{}' has overflowed its stack", name.unwrap_or("<unknown>"));
}

// get_task_info is called from an exception / signal handler.
// It returns the guard page of the current task or 0 if that
// guard page doesn't exist. None is returned if there's currently
// no local task.
#[cfg(any(windows, target_os = "linux", target_os = "macos"))]
unsafe fn get_task_guard_page() -> Option<uint> {
    let task: Option<*mut Task> = Local::try_unsafe_borrow();
    task.map(|task| (&*task).stack_guard().unwrap_or(0))
}

#[cfg(windows)]
#[allow(non_snake_case)]
mod imp {
    use core::ptr;
    use core::mem;
    use libc;
    use libc::types::os::arch::extra::{LPVOID, DWORD, LONG, BOOL};
    use stack;
    use super::{Handler, get_task_guard_page, report};

    // This is initialized in init() and only read from after
    static mut PAGE_SIZE: uint = 0;

    #[no_stack_check]
    extern "system" fn vectored_handler(ExceptionInfo: *mut EXCEPTION_POINTERS) -> LONG {
        unsafe {
            let rec = &(*(*ExceptionInfo).ExceptionRecord);
            let code = rec.ExceptionCode;

            if code != EXCEPTION_STACK_OVERFLOW {
                return EXCEPTION_CONTINUE_SEARCH;
            }

            // We're calling into functions with stack checks,
            // however stack checks by limit should be disabled on Windows
            stack::record_sp_limit(0);

            if get_task_guard_page().is_some() {
               report();
            }

            EXCEPTION_CONTINUE_SEARCH
        }
    }

    pub unsafe fn init() {
        let mut info = mem::zeroed();
        libc::GetSystemInfo(&mut info);
        PAGE_SIZE = info.dwPageSize as uint;

        if AddVectoredExceptionHandler(0, vectored_handler) == ptr::null_mut() {
            panic!("failed to install exception handler");
        }

        mem::forget(make_handler());
    }

    pub unsafe fn cleanup() {
    }

    pub unsafe fn make_handler() -> Handler {
        if SetThreadStackGuarantee(&mut 0x5000) == 0 {
            panic!("failed to reserve stack space for exception handling");
        }

        super::Handler { _data: 0i as *mut libc::c_void }
    }

    pub unsafe fn drop_handler(_handler: &mut Handler) {
    }

    pub struct EXCEPTION_RECORD {
        pub ExceptionCode: DWORD,
        pub ExceptionFlags: DWORD,
        pub ExceptionRecord: *mut EXCEPTION_RECORD,
        pub ExceptionAddress: LPVOID,
        pub NumberParameters: DWORD,
        pub ExceptionInformation: [LPVOID, ..EXCEPTION_MAXIMUM_PARAMETERS]
    }

    pub struct EXCEPTION_POINTERS {
        pub ExceptionRecord: *mut EXCEPTION_RECORD,
        pub ContextRecord: LPVOID
    }

    pub type PVECTORED_EXCEPTION_HANDLER = extern "system"
            fn(ExceptionInfo: *mut EXCEPTION_POINTERS) -> LONG;

    pub type ULONG = libc::c_ulong;

    const EXCEPTION_CONTINUE_SEARCH: LONG = 0;
    const EXCEPTION_MAXIMUM_PARAMETERS: uint = 15;
    const EXCEPTION_STACK_OVERFLOW: DWORD = 0xc00000fd;

    extern "system" {
        fn AddVectoredExceptionHandler(FirstHandler: ULONG,
                                       VectoredHandler: PVECTORED_EXCEPTION_HANDLER)
                                      -> LPVOID;
        fn SetThreadStackGuarantee(StackSizeInBytes: *mut ULONG) -> BOOL;
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod imp {
    use core::prelude::*;
    use stack;

    use super::{Handler, get_task_guard_page, report};
    use core::mem;
    use core::ptr;
    use core::intrinsics;
    use self::signal::{siginfo, sigaction, SIGBUS, SIG_DFL,
                       SA_SIGINFO, SA_ONSTACK, sigaltstack,
                       SIGSTKSZ};
    use libc;
    use libc::funcs::posix88::mman::{mmap, munmap};
    use libc::consts::os::posix88::{SIGSEGV,
                                    PROT_READ,
                                    PROT_WRITE,
                                    MAP_PRIVATE,
                                    MAP_ANON,
                                    MAP_FAILED};


    // This is initialized in init() and only read from after
    static mut PAGE_SIZE: uint = 0;

    #[no_stack_check]
    unsafe extern fn signal_handler(signum: libc::c_int,
                                     info: *mut siginfo,
                                     _data: *mut libc::c_void) {

        // We can not return from a SIGSEGV or SIGBUS signal.
        // See: https://www.gnu.org/software/libc/manual/html_node/Handler-Returns.html

        unsafe fn term(signum: libc::c_int) -> ! {
            use core::mem::transmute;

            signal(signum, transmute(SIG_DFL));
            raise(signum);
            intrinsics::abort();
        }

        // We're calling into functions with stack checks
        stack::record_sp_limit(0);

        match get_task_guard_page() {
            Some(guard) => {
                let addr = (*info).si_addr as uint;

                if guard == 0 || addr < guard - PAGE_SIZE || addr >= guard {
                    term(signum);
                }

                report();

                intrinsics::abort()
            }
            None => term(signum)
        }
    }

    static mut MAIN_ALTSTACK: *mut libc::c_void = 0 as *mut libc::c_void;

    pub unsafe fn init() {
        let psize = libc::sysconf(libc::consts::os::sysconf::_SC_PAGESIZE);
        if psize == -1 {
            panic!("failed to get page size");
        }

        PAGE_SIZE = psize as uint;

        let mut action: sigaction = mem::zeroed();
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        action.sa_sigaction = signal_handler as sighandler_t;
        sigaction(SIGSEGV, &action, ptr::null_mut());
        sigaction(SIGBUS, &action, ptr::null_mut());

        let handler = make_handler();
        MAIN_ALTSTACK = handler._data;
        mem::forget(handler);
    }

    pub unsafe fn cleanup() {
        Handler { _data: MAIN_ALTSTACK };
    }

    pub unsafe fn make_handler() -> Handler {
        let alt_stack = mmap(ptr::null_mut(),
                             signal::SIGSTKSZ,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANON,
                             -1,
                             0);
        if alt_stack == MAP_FAILED {
            panic!("failed to allocate an alternative stack");
        }

        let mut stack: sigaltstack = mem::zeroed();

        stack.ss_sp = alt_stack;
        stack.ss_flags = 0;
        stack.ss_size = SIGSTKSZ;

        sigaltstack(&stack, ptr::null_mut());

        Handler { _data: alt_stack }
    }

    pub unsafe fn drop_handler(handler: &mut Handler) {
        munmap(handler._data, SIGSTKSZ);
    }

    type sighandler_t = *mut libc::c_void;

    #[cfg(any(all(target_os = "linux", target_arch = "x86"), // may not match
              all(target_os = "linux", target_arch = "x86_64"),
              all(target_os = "linux", target_arch = "arm"), // may not match
              all(target_os = "linux", target_arch = "mips"), // may not match
              target_os = "android"))] // may not match
    mod signal {
        use libc;
        use super::sighandler_t;

        pub static SA_ONSTACK: libc::c_int = 0x08000000;
        pub static SA_SIGINFO: libc::c_int = 0x00000004;
        pub static SIGBUS: libc::c_int = 7;

        pub static SIGSTKSZ: libc::size_t = 8192;

        pub static SIG_DFL: sighandler_t = 0i as sighandler_t;

        // This definition is not as accurate as it could be, {si_addr} is
        // actually a giant union. Currently we're only interested in that field,
        // however.
        #[repr(C)]
        pub struct siginfo {
            si_signo: libc::c_int,
            si_errno: libc::c_int,
            si_code: libc::c_int,
            pub si_addr: *mut libc::c_void
        }

        #[repr(C)]
        pub struct sigaction {
            pub sa_sigaction: sighandler_t,
            pub sa_mask: sigset_t,
            pub sa_flags: libc::c_int,
            sa_restorer: *mut libc::c_void,
        }

        #[cfg(target_word_size = "32")]
        #[repr(C)]
        pub struct sigset_t {
            __val: [libc::c_ulong, ..32],
        }
        #[cfg(target_word_size = "64")]
        #[repr(C)]
        pub struct sigset_t {
            __val: [libc::c_ulong, ..16],
        }

        #[repr(C)]
        pub struct sigaltstack {
            pub ss_sp: *mut libc::c_void,
            pub ss_flags: libc::c_int,
            pub ss_size: libc::size_t
        }

    }

    #[cfg(target_os = "macos")]
    mod signal {
        use libc;
        use super::sighandler_t;

        pub const SA_ONSTACK: libc::c_int = 0x0001;
        pub const SA_SIGINFO: libc::c_int = 0x0040;
        pub const SIGBUS: libc::c_int = 10;

        pub const SIGSTKSZ: libc::size_t = 131072;

        pub const SIG_DFL: sighandler_t = 0i as sighandler_t;

        pub type sigset_t = u32;

        // This structure has more fields, but we're not all that interested in
        // them.
        #[repr(C)]
        pub struct siginfo {
            pub si_signo: libc::c_int,
            pub si_errno: libc::c_int,
            pub si_code: libc::c_int,
            pub pid: libc::pid_t,
            pub uid: libc::uid_t,
            pub status: libc::c_int,
            pub si_addr: *mut libc::c_void
        }

        #[repr(C)]
        pub struct sigaltstack {
            pub ss_sp: *mut libc::c_void,
            pub ss_size: libc::size_t,
            pub ss_flags: libc::c_int
        }

        #[repr(C)]
        pub struct sigaction {
            pub sa_sigaction: sighandler_t,
            pub sa_mask: sigset_t,
            pub sa_flags: libc::c_int,
        }
    }

    extern {
        pub fn signal(signum: libc::c_int, handler: sighandler_t) -> sighandler_t;
        pub fn raise(signum: libc::c_int) -> libc::c_int;

        pub fn sigaction(signum: libc::c_int,
                         act: *const sigaction,
                         oldact: *mut sigaction) -> libc::c_int;

        pub fn sigaltstack(ss: *const sigaltstack,
                           oss: *mut sigaltstack) -> libc::c_int;
    }
}

#[cfg(not(any(target_os = "linux",
              target_os = "macos",
              windows)))]
mod imp {
    use libc;

    pub unsafe fn init() {
    }

    pub unsafe fn cleanup() {
    }

    pub unsafe fn make_handler() -> super::Handler {
        super::Handler { _data: 0i as *mut libc::c_void }
    }

    pub unsafe fn drop_handler(_handler: &mut super::Handler) {
    }
}
