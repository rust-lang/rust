cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "dragonfly",
        target_os = "fuchsia",
        all(target_family = "wasm", target_feature = "atomics"),
        target_os = "hermit",
    ))] {
        mod futex;
        pub use futex::Condvar;
    } else if #[cfg(target_family = "unix")] {
        mod pthread;
        pub use pthread::Condvar;
    } else if #[cfg(target_os = "windows")] {
        mod windows;
        pub use windows::Condvar;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use sgx::Condvar;
    } else if #[cfg(target_os = "solid_asp3")] {
        mod itron;
        pub use itron::Condvar;
    } else if #[cfg(target_os = "teeos")] {
        mod teeos;
        pub use teeos::Condvar;
    } else if #[cfg(target_os = "xous")] {
        mod xous;
        pub use xous::Condvar;
    } else {
        mod no_threads;
        pub use no_threads::Condvar;
    }
}
