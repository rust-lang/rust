cfg_if::cfg_if! {
    if #[cfg(any(
        all(target_os = "windows", not(target_vendor = "win7")),
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
        pub use futex::RwLock;
    } else if #[cfg(any(
        target_family = "unix",
        all(target_os = "windows", target_vendor = "win7"),
        all(target_vendor = "fortanix", target_env = "sgx"),
        target_os = "xous",
    ))] {
        mod queue;
        pub use queue::RwLock;
    } else if #[cfg(target_os = "solid_asp3")] {
        mod solid;
        pub use solid::RwLock;
    } else if #[cfg(target_os = "teeos")] {
        mod teeos;
        pub use teeos::RwLock;
    } else {
        mod no_threads;
        pub use no_threads::RwLock;
    }
}
