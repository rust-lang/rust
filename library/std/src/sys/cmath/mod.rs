#![cfg(not(test))]

cfg_if::cfg_if! {
    if #[cfg(target_os = "windows")] {
        mod windows;
        pub use windows::*;
    } else {
        mod builtins;
        pub use builtins::*;
    }
}
