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
    // The wasi-libc based futex is new enough that it's not present in older
    // wasi-libc builds. For now that means it's only required on wasip3 (which
    // requires a newer wasi-libc anyway). In the future this'll probably switch to
    // unconditionally using `wasilibc` as the implementation for all WASI
    // targets (and switching all synchronization primitives to the futex version).
    all(target_os = "wasi", target_env = "p3") => {
        mod wasilibc;
        pub use wasilibc::*;
    }
    all(target_family = "wasm", target_feature = "atomics") => {
        mod wasm;
        pub use wasm::*;
    }
    target_os = "motor" => {
        pub use moto_rt::futex::*;
    }
    target_os = "vexos" => {
        mod vexos;
        pub use vexos::*;
    }
    _ => {}
}
