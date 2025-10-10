cfg_select! {
    any(
        all(target_os = "windows", not(target_vendor = "win7")),
        target_os = "linux",
        target_os = "android",
        all(target_arch = "wasm32", target_feature = "atomics"),
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "dragonfly",
        target_os = "fuchsia",
        target_os = "hermit",
    ) => {
        mod futex;
        pub use futex::Parker;
    }
    any(
        target_os = "netbsd",
        all(target_vendor = "fortanix", target_env = "sgx"),
        target_os = "solid_asp3",
    ) => {
        mod id;
        pub use id::Parker;
    }
    target_vendor = "win7" => {
        mod windows7;
        pub use windows7::Parker;
    }
    all(target_vendor = "apple", not(miri)) => {
        // Doesn't work in Miri, see <https://github.com/rust-lang/miri/issues/2589>.
        mod darwin;
        pub use darwin::Parker;
    }
    target_os = "xous" => {
        mod xous;
        pub use xous::Parker;
    }
    target_family = "unix" => {
        mod pthread;
        pub use pthread::Parker;
    }
    _ => {
        mod unsupported;
        pub use unsupported::Parker;
    }
}
