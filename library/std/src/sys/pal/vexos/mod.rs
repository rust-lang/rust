pub mod alloc;
#[path = "../unsupported/args.rs"]
pub mod args;
pub mod env;
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
pub mod stdio;
#[path = "../unsupported/thread.rs"]
pub mod thread;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
pub mod time;

use crate::{arch::asm, ptr::{self, addr_of_mut}};
use crate::hash::{DefaultHasher, Hasher};

#[cfg(not(test))]
#[no_mangle]
#[link_section = ".boot"]
pub unsafe extern "C" fn _start() -> ! {
    extern "C" {
        static mut __bss_start: u8;
        static mut __bss_end: u8;

        fn main() -> i32;
    }

    // Setup the stack
    asm!("ldr sp, =__stack_top", options(nostack));

    // vexOS doesn't explicitly clean out .bss, so as a sanity
    // check we'll fill it with zeroes.
    ptr::slice_from_raw_parts_mut(
        addr_of_mut!(__bss_start),
        addr_of_mut!(__bss_end).offset_from(addr_of_mut!(__bss_start)) as usize,
    )
    .as_mut()
    .unwrap_unchecked()
    .fill(0);

    main();

    abort_internal()
}

#[link_section = ".code_signature"]
#[linkage = "weak"]
#[used]
static CODE_SIGNATURE: vex_sdk::vcodesig = vex_sdk::vcodesig {
    magic: u32::from_le_bytes(*b"XVX5"),
    r#type: 2,
    owner: 0,
    options: 0,
};

// This function is needed by the panic runtime. The symbol is named in
// pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
#[no_mangle]
// NB. used by both libunwind and libpanic_abort
pub extern "C" fn __rust_abort() -> ! {
    abort_internal()
}

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(_argc: isize, _argv: *const *const u8, _sigpipe: u8) {}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
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
        vex_sdk::vexTasksRun();
        vex_sdk::vexSystemExitRequest();
    }

    loop {
        crate::hint::spin_loop()
    }
}

fn hash_time() -> u64 {
    let mut hasher = DefaultHasher::new();
    // The closest we can get to a random number is the time since program start
    let time = unsafe {
        vex_sdk::vexSystemHighResTimeGet()
    };
    hasher.write_u64(time);
    hasher.finish()
}

pub fn hashmap_random_keys() -> (u64, u64) {
    let key1 = hash_time();
    let key2 = hash_time();
    (key1, key2)
}
