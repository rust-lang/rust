cfg_select! {
    any(
        all(target_os = "windows", not(target_vendor="win7")),
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "dragonfly",
        target_os = "fuchsia",
        all(target_family = "wasm", target_feature = "atomics"),
        target_os = "hermit",
    ) => {
        mod futex;
        pub use futex::Condvar;
    }
    any(
        target_family = "unix",
        target_os = "teeos",
    ) => {
        mod pthread;
        pub use pthread::Condvar;
    }
    all(target_os = "windows", target_vendor = "win7") => {
        mod windows7;
        pub use windows7::Condvar;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        pub use sgx::Condvar;
    }
    target_os = "solid_asp3" => {
        mod itron;
        pub use itron::Condvar;
    }
    target_os = "xous" => {
        mod xous;
        pub use xous::Condvar;
    }
    _ => {
        mod no_threads;
        pub use no_threads::Condvar;
    }
}
