#![feature(cfg_target_thread_local)]

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))]
mod tests;

mod dynamic_tests;
