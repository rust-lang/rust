#![allow(unused_crate_dependencies)]

use std::error::Error;

use rustc_thread_pool::ThreadPoolBuilder;

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn double_init_fail() {
    let result1 = ThreadPoolBuilder::new().build_global();
    assert!(result1.is_ok());
    let err = ThreadPoolBuilder::new().build_global().unwrap_err();
    assert!(err.source().is_none());
    assert_eq!(err.to_string(), "The global thread pool has already been initialized.",);
}
