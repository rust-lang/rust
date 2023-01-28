//! Helper module which helps to determine amount of threads to be used
//! during tests execution.
use std::{env, num::NonZeroUsize, thread};

pub fn get_concurrency() -> usize {
    rust_test_threads_from_env()
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

pub fn supports_threads() -> bool {
    if cfg!(target_os = "emscripten") || cfg!(target_family = "wasm") {
        return false;
    }

    // Accommodate libraries that may rely on shared thread-local storage (e.g.
    // integrating with old C libraries).
    if let Some(1) = rust_test_threads_from_env() {
        return false;
    }

    true
}

fn rust_test_threads_from_env() -> Option<usize> {
    let value = env::var("RUST_TEST_THREADS").ok()?;

    if let Ok(value) = value.parse::<NonZeroUsize>() {
        Some(value.get())
    } else {
        panic!("RUST_TEST_THREADS is `{value}`, should be a positive integer.", value = value)
    }
}
