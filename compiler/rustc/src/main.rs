// We need this feature as it changes `dylib` linking behavior and allows us to link to `rustc_driver`.
#![feature(rustc_private)]
// Several crates are depended upon but unused so that they are present in the sysroot
#![expect(unused_crate_dependencies)]

use std::process::ExitCode;

// A note about jemalloc: rustc uses jemalloc when built for CI and
// distribution. The obvious way to do this is with the `#[global_allocator]`
// mechanism. However, that would not affect LLVM's C / C++ allocations and we also want
// to use a single allocator in the process to reduce memory usage.
//
// Instead, we use a lower-level mechanism, namely the
// `"override_allocator_on_supported_platforms"` Cargo feature of jemalloc-sys.
//
// This makes jemalloc-sys override the libc/system allocator's implementation
// of `malloc`, `free`, etc.. This means that Rust's `System` allocator, which
// calls `libc::malloc()` et al., is actually calling into jemalloc.
//
// This override happens for the entire process, ensuring that there's no mixup
// of C allocators across dylibs / binaries, notably the rustc <-> llvm boundary.
//
// A consequence of not using `GlobalAlloc` (and the `tikv-jemallocator` crate
// provides an impl of that trait, which is called `Jemalloc`) is that we
// cannot use the sized deallocation APIs (`sdallocx`) that jemalloc provides.
// It's unclear how much performance is lost because of this.
//
// NOTE: if you are reading this comment because you want to set a custom `global_allocator` for
// benchmarking, consider using the benchmarks in the `rustc-perf` collector suite instead:
// https://github.com/rust-lang/rustc-perf/blob/master/collector/README.md#profiling
//
// NOTE: if you are reading this comment because you want to replace jemalloc with another allocator
// to compare their performance, see
// https://github.com/rust-lang/rust/commit/b90cfc887c31c3e7a9e6d462e2464db1fe506175#diff-43914724af6e464c1da2171e4a9b6c7e607d5bc1203fa95c0ab85be4122605ef
// for an example of how to do so.
rustc_driver::override_c_allocator_in_binary!();

fn main() -> ExitCode {
    rustc_driver::main()
}
