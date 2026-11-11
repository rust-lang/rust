/// This macro overrides the C allocator (i.e., `malloc`) in final binaries by linking
/// jemalloc with the override feature enabled. The C allocator is used by the default
/// Rust allocator (`alloc::System`) on Unix targets but not on Windows targets.
#[cfg(feature = "jemalloc")]
#[macro_export]
macro_rules! override_c_allocator_in_binary {
    () => {
        // NOTE: even though Cargo passes `--extern` for this in rustc-main, the crate still has
        // to be named for the compiler to see the `#[used]` inside it, see
        // <https://github.com/rust-lang/rust/issues/64402>.
        //
        // FIXME(madsmtm): for the rustc-private tools this is loaded from the sysroot that was
        // built with the other `rustc` crates, instead of via Cargo as you'd normally do. This is
        // currently needed for LTO due to <https://github.com/rust-lang/cc-rs/issues/1613>.
        extern crate tikv_jemalloc_sys as _;
    };
}

/// This macro does nothing when no allocator features are enabled.
#[cfg(not(feature = "jemalloc"))]
#[macro_export]
macro_rules! override_c_allocator_in_binary {
    () => {};
}
