cfg_select! {
    any(
        target_os = "linux",
        target_os = "android",
        all(target_os = "emscripten", target_feature = "atomics"),
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "dragonfly",
        target_os = "fuchsia",
    ) => {
        mod unix;
        pub use unix::*;
    }
    all(target_os = "windows", not(target_vendor = "win7")) => {
        mod windows;
        pub use windows::*;
    }
    target_os = "hermit" => {
        mod hermit;
        pub use hermit::*;
    }
    all(target_family = "wasm", target_feature = "atomics") => {
        mod wasm;
        pub use wasm::*;
    }
    target_os = "motor" => {
        pub use moto_rt::futex::*;
    }
    _ => {}
}
