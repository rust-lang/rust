cfg_if::cfg_if! {
    if #[cfg(any(
        all(target_os = "windows", not(target_vendor="win7")),
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
    } else if #[cfg(any(
        all(target_os = "windows", target_vendor = "win7"),
        target_os = "netbsd",
        all(target_vendor = "fortanix", target_env = "sgx"),
        target_os = "teeos",
        target_os = "xous",
    ))] {
        mod queue;
        pub use queue::Condvar;
    } else if #[cfg(target_family = "unix")] {
        mod pthread;
        pub use pthread::Condvar;
    } else if #[cfg(target_os = "solid_asp3")] {
        mod itron;
        pub use itron::Condvar;
    } else {
        mod no_threads;
        pub use no_threads::Condvar;
    }
}
