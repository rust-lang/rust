//! Thread parking on systems without futex support.

#![cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics"),
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly",
    target_os = "fuchsia",
)))]

cfg_match! {
    cfg(all(
        any(
            target_os = "macos",
            target_os = "ios",
            target_os = "watchos",
            target_os = "tvos",
        ),
        not(miri),
    )) => {
        mod darwin;
        pub use darwin::Parker;
    }
    cfg(target_os = "netbsd") => {
        mod netbsd;
        pub use netbsd::{current, park, park_timeout, unpark, ThreadId};
    }
    _ => {
        mod pthread;
        pub use pthread::Parker;
    }
}
