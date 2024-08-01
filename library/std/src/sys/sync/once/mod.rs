// A "once" is a relatively simple primitive, and it's also typically provided
// by the OS as well (see `pthread_once` or `InitOnceExecuteOnce`). The OS
// primitives, however, tend to have surprising restrictions, such as the Unix
// one doesn't allow an argument to be passed to the function.
//
// As a result, we end up implementing it ourselves in the standard library.
// This also gives us the opportunity to optimize the implementation a bit which
// should help the fast path on call sites.

cfg_if::cfg_if! {
    if #[cfg(any(
        all(target_os = "windows", not(target_vendor="win7")),
        target_os = "linux",
        target_os = "android",
        all(target_arch = "wasm32", target_feature = "atomics"),
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "dragonfly",
        target_os = "fuchsia",
        target_os = "hermit",
    ))] {
        mod futex;
        pub use futex::{Once, OnceState};
    } else if #[cfg(any(
        windows,
        target_family = "unix",
        all(target_vendor = "fortanix", target_env = "sgx"),
        target_os = "solid_asp3",
        target_os = "xous",
    ))] {
        mod queue;
        pub use queue::{Once, OnceState};
    } else {
        mod no_threads;
        pub use no_threads::{Once, OnceState};
    }
}
