//! A collection of tools for profiling rust-analyzer.

mod memory_usage;
#[cfg(feature = "cpu_profiler")]
mod google_cpu_profiler;
mod hprof;
mod tree;

use std::cell::RefCell;

pub use crate::{
    hprof::{init, init_from, profile},
    memory_usage::{Bytes, MemoryUsage},
};

// We use jemalloc mainly to get heap usage statistics, actual performance
// difference is not measures.
#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// Prints backtrace to stderr, useful for debugging.
#[cfg(feature = "backtrace")]
pub fn print_backtrace() {
    let bt = backtrace::Backtrace::new();
    eprintln!("{:?}", bt);
}
#[cfg(not(feature = "backtrace"))]
pub fn print_backtrace() {
    eprintln!(
        r#"enable the backtrace feature:
    ra_prof = {{ path = "../ra_prof", features = [ "backtrace"] }}
"#
    );
}

thread_local!(static IN_SCOPE: RefCell<bool> = RefCell::new(false));

/// Allows to check if the current code is withing some dynamic scope, can be
/// useful during debugging to figure out why a function is called.
pub struct Scope {
    prev: bool,
}

impl Scope {
    pub fn enter() -> Scope {
        let prev = IN_SCOPE.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), true));
        Scope { prev }
    }
    pub fn is_active() -> bool {
        IN_SCOPE.with(|slot| *slot.borrow())
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        IN_SCOPE.with(|slot| *slot.borrow_mut() = self.prev);
    }
}

/// A wrapper around google_cpu_profiler.
///
/// Usage:
/// 1. Install gpref_tools (https://github.com/gperftools/gperftools), probably packaged with your Linux distro.
/// 2. Build with `cpu_profiler` feature.
/// 3. Tun the code, the *raw* output would be in the `./out.profile` file.
/// 4. Install pprof for visualization (https://github.com/google/pprof).
/// 5. Use something like `pprof -svg target/release/rust-analyzer ./out.profile` to see the results.
///
/// For example, here's how I run profiling on NixOS:
///
/// ```bash
/// $ nix-shell -p gperftools --run \
///     'cargo run --release -p rust-analyzer -- parse < ~/projects/rustbench/parser.rs > /dev/null'
/// ```
#[derive(Debug)]
pub struct CpuProfiler {
    _private: (),
}

pub fn cpu_profiler() -> CpuProfiler {
    #[cfg(feature = "cpu_profiler")]
    {
        google_cpu_profiler::start("./out.profile".as_ref())
    }

    #[cfg(not(feature = "cpu_profiler"))]
    {
        eprintln!("cpu_profiler feature is disabled")
    }

    CpuProfiler { _private: () }
}

impl Drop for CpuProfiler {
    fn drop(&mut self) {
        #[cfg(feature = "cpu_profiler")]
        {
            google_cpu_profiler::stop()
        }
    }
}

pub fn memory_usage() -> MemoryUsage {
    MemoryUsage::current()
}
