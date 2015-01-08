// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use core::prelude::*;
use self::imp::{make_handler, drop_handler};

pub use self::imp::{init, cleanup};

pub struct Handler {
    _data: *mut libc::c_void
}

impl Handler {
    pub unsafe fn new() -> Handler {
        make_handler()
    }
}

impl Drop for Handler {
    fn drop(&mut self) {
        unsafe {
            drop_handler(self);
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod imp {
    use core::prelude::*;
    use sys_common::stack;

    use super::Handler;
    use rt::util::report_overflow;
    use mem;
    use ptr;
    use intrinsics;
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

    use sys_common::thread_info;


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

        let guard = thread_info::stack_guard();
        let addr = (*info).si_addr as uint;

        if guard == 0 || addr < guard - PAGE_SIZE || addr >= guard {
            term(signum);
        }

        report_overflow();

        intrinsics::abort()
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
              all(target_os = "linux", target_arch = "aarch64"),
              all(target_os = "linux", target_arch = "mips"), // may not match
              all(target_os = "linux", target_arch = "mipsel"), // may not match
              target_os = "android"))] // may not match
    mod signal {
        use libc;
        use super::sighandler_t;

        pub static SA_ONSTACK: libc::c_int = 0x08000000;
        pub static SA_SIGINFO: libc::c_int = 0x00000004;
        pub static SIGBUS: libc::c_int = 7;

        pub static SIGSTKSZ: libc::size_t = 8192;

        pub const SIG_DFL: sighandler_t = 0i as sighandler_t;

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

        #[cfg(any(all(stage0, target_word_size = "32"),
                  all(not(stage0), target_pointer_width = "32")))]
        #[repr(C)]
        pub struct sigset_t {
            __val: [libc::c_ulong; 32],
        }
        #[cfg(any(all(stage0, target_word_size = "64"),
                  all(not(stage0), target_pointer_width = "64")))]
        #[repr(C)]
        pub struct sigset_t {
            __val: [libc::c_ulong; 16],
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
              target_os = "macos")))]
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
