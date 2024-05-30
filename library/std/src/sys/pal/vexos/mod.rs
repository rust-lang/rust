pub mod alloc;
#[path = "../unsupported/args.rs"]
pub mod args;
#[path = "../unsupported/env.rs"]
pub mod env;
#[path = "../unsupported/fs.rs"]
pub mod fs;
#[path = "../unsupported/io.rs"]
pub mod io;
#[path = "../unsupported/net.rs"]
pub mod net;
#[path = "../unsupported/os.rs"]
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
#[path = "../unsupported/stdio.rs"]
pub mod stdio;
#[path = "../unsupported/thread.rs"]
pub mod thread;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
#[path = "../unsupported/time.rs"]
pub mod time;

use crate::arch::asm;

#[cfg(not(test))]
#[no_mangle]
#[link_section = ".text.boot"]
pub unsafe extern "C" fn _start() -> ! {
    extern "C" {
        fn main() -> i32;
    }

    asm!("ldr sp, =__stack_top", options(nostack));

    main();

    abort_internal()
}

// This function is needed by the panic runtime. The symbol is named in
// pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
#[no_mangle]
// NB. used by both libunwind and libpanic_abort
pub extern "C" fn __rust_abort() -> ! {
    abort_internal()
}

pub unsafe fn init(_argc: isize, _argv: *const *const u8, _sigpipe: u8) {}

pub unsafe fn cleanup() {}

pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> crate::io::Error {
    crate::io::Error::UNSUPPORTED_PLATFORM
}

pub fn is_interrupted(_code: i32) -> bool {
    false
}

pub fn decode_error_kind(_code: i32) -> crate::io::ErrorKind {
    crate::io::ErrorKind::Uncategorized
}

pub fn abort_internal() -> ! {
    unsafe {
        vex_sdk::vexSystemExitRequest();
    }

    loop {
        crate::hint::spin_loop()
    }
}

pub fn hashmap_random_keys() -> (u64, u64) {
    (1, 2)
}
