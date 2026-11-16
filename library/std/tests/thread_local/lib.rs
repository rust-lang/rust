#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))]
mod tests;

mod dynamic_tests;
