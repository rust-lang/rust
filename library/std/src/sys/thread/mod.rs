cfg_select! {
    target_os = "hermit" => {
        mod hermit;
        pub use hermit::{Thread, available_parallelism, sleep, yield_now, DEFAULT_MIN_STACK_SIZE};
        #[expect(dead_code)]
        mod unsupported;
        pub use unsupported::{current_os_id, set_name};
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        pub use sgx::{Thread, current_os_id, sleep, yield_now, DEFAULT_MIN_STACK_SIZE};

        // SGX should protect in-enclave data from outside attackers, so there
        // must not be any data leakage to the OS, particularly no 1-1 mapping
        // between SGX thread names and OS thread names. Hence `set_name` is
        // intentionally a no-op.
        //
        // Note that the internally visible SGX thread name is already provided
        // by the platform-agnostic Rust thread code. This can be observed in
        // the [`std::thread::tests::test_named_thread`] test, which succeeds
        // as-is with the SGX target.
        #[expect(dead_code)]
        mod unsupported;
        pub use unsupported::{available_parallelism, set_name};
    }
    target_os = "solid_asp3" => {
        mod solid;
        pub use solid::{Thread, sleep, yield_now, DEFAULT_MIN_STACK_SIZE};
        #[expect(dead_code)]
        mod unsupported;
        pub use unsupported::{available_parallelism, current_os_id, set_name};
    }
    target_os = "teeos" => {
        mod teeos;
        pub use teeos::{Thread, sleep, yield_now, DEFAULT_MIN_STACK_SIZE};
        #[expect(dead_code)]
        mod unsupported;
        pub use unsupported::{available_parallelism, current_os_id, set_name};
    }
    target_os = "uefi" => {
        mod uefi;
        pub use uefi::{available_parallelism, sleep};
        #[expect(dead_code)]
        mod unsupported;
        pub use unsupported::{Thread, current_os_id, set_name, yield_now, DEFAULT_MIN_STACK_SIZE};
    }
    target_family = "unix" => {
        mod unix;
        pub use unix::{Thread, available_parallelism, current_os_id, sleep, yield_now, DEFAULT_MIN_STACK_SIZE};
        #[cfg(not(any(
            target_env = "newlib",
            target_os = "l4re",
            target_os = "emscripten",
            target_os = "redox",
            target_os = "hurd",
            target_os = "aix",
        )))]
        pub use unix::set_name;
        #[cfg(any(
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "linux",
            target_os = "android",
            target_os = "solaris",
            target_os = "illumos",
            target_os = "dragonfly",
            target_os = "hurd",
            target_os = "fuchsia",
            target_os = "vxworks",
        ))]
        pub use unix::sleep_until;
        #[expect(dead_code)]
        mod unsupported;
        #[cfg(any(
            target_env = "newlib",
            target_os = "l4re",
            target_os = "emscripten",
            target_os = "redox",
            target_os = "hurd",
            target_os = "aix",
        ))]
        pub use unsupported::set_name;
    }
    target_os = "vexos" => {
        mod vexos;
        pub use vexos::{sleep, yield_now};
        #[expect(dead_code)]
        mod unsupported;
        pub use unsupported::{Thread, available_parallelism, current_os_id, set_name, DEFAULT_MIN_STACK_SIZE};
    }
    all(target_os = "wasi", target_env = "p1") => {
        mod wasip1;
        pub use wasip1::{DEFAULT_MIN_STACK_SIZE, sleep, yield_now};
        #[cfg(target_feature = "atomics")]
        pub use wasip1::{Thread, available_parallelism};
        #[expect(dead_code)]
        mod unsupported;
        pub use unsupported::{current_os_id, set_name};
        #[cfg(not(target_feature = "atomics"))]
        pub use unsupported::{Thread, available_parallelism};
    }
    all(target_os = "wasi", target_env = "p2") => {
        mod wasip2;
        pub use wasip2::{sleep, sleep_until};
        #[expect(dead_code)]
        mod unsupported;
        // Note that unlike WASIp1 even if the wasm `atomics` feature is enabled
        // there is no support for threads, not even experimentally, not even in
        // wasi-libc. Thus this is unconditionally unsupported.
        pub use unsupported::{Thread, available_parallelism, current_os_id, set_name, yield_now, DEFAULT_MIN_STACK_SIZE};
    }
    all(target_family = "wasm", target_feature = "atomics") => {
        mod wasm;
        pub use wasm::sleep;

        #[expect(dead_code)]
        mod unsupported;
        pub use unsupported::{Thread, available_parallelism, current_os_id, set_name, yield_now, DEFAULT_MIN_STACK_SIZE};
    }
    target_os = "windows" => {
        mod windows;
        pub use windows::{Thread, available_parallelism, current_os_id, set_name, set_name_wide, sleep, yield_now, DEFAULT_MIN_STACK_SIZE};
    }
    target_os = "xous" => {
        mod xous;
        pub use xous::{Thread, available_parallelism, sleep, yield_now, DEFAULT_MIN_STACK_SIZE};

        #[expect(dead_code)]
        mod unsupported;
        pub use unsupported::{current_os_id, set_name};
    }
    _ => {
        mod unsupported;
        pub use unsupported::{Thread, available_parallelism, current_os_id, set_name, sleep, yield_now, DEFAULT_MIN_STACK_SIZE};
    }
}

#[cfg(not(any(
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "linux",
    target_os = "android",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "dragonfly",
    target_os = "hurd",
    target_os = "fuchsia",
    target_os = "vxworks",
    all(target_os = "wasi", target_env = "p2"),
)))]
pub fn sleep_until(deadline: crate::time::Instant) {
    use crate::time::Instant;

    let now = Instant::now();

    if let Some(delay) = deadline.checked_duration_since(now) {
        sleep(delay);
    }
}
