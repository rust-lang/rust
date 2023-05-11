//! A collection of tools for profiling rust-analyzer.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod stop_watch;
mod memory_usage;
#[cfg(feature = "cpu_profiler")]
mod google_cpu_profiler;
mod hprof;
mod tree;

use std::cell::RefCell;

pub use crate::{
    hprof::{heartbeat, heartbeat_span, init, init_from, span},
    memory_usage::{Bytes, MemoryUsage},
    stop_watch::{StopWatch, StopWatchSpan},
};

pub use countme;
/// Include `_c: Count<Self>` field in important structs to count them.
///
/// To view the counts, run with `RA_COUNT=1`. The overhead of disabled count is
/// almost zero.
pub use countme::Count;

thread_local!(static IN_SCOPE: RefCell<bool> = RefCell::new(false));

/// Allows to check if the current code is within some dynamic scope, can be
/// useful during debugging to figure out why a function is called.
pub struct Scope {
    prev: bool,
}

impl Scope {
    #[must_use]
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
/// 1. Install gpref_tools (<https://github.com/gperftools/gperftools>), probably packaged with your Linux distro.
/// 2. Build with `cpu_profiler` feature.
/// 3. Run the code, the *raw* output would be in the `./out.profile` file.
/// 4. Install pprof for visualization (<https://github.com/google/pprof>).
/// 5. Bump sampling frequency to once per ms: `export CPUPROFILE_FREQUENCY=1000`
/// 6. Use something like `pprof -svg target/release/rust-analyzer ./out.profile` to see the results.
///
/// For example, here's how I run profiling on NixOS:
///
/// ```bash
/// $ bat -p shell.nix
/// with import <nixpkgs> {};
/// mkShell {
///   buildInputs = [ gperftools ];
///   shellHook = ''
///     export LD_LIBRARY_PATH="${gperftools}/lib:"
///   '';
/// }
/// $ set -x CPUPROFILE_FREQUENCY 1000
/// $ nix-shell --run 'cargo test --release --package rust-analyzer --lib -- benchmarks::benchmark_integrated_highlighting --exact --nocapture'
/// $ pprof -svg target/release/deps/rust_analyzer-8739592dc93d63cb crates/rust-analyzer/out.profile > profile.svg
/// ```
///
/// See this diff for how to profile completions:
///
/// <https://github.com/rust-lang/rust-analyzer/pull/5306>
#[derive(Debug)]
pub struct CpuSpan {
    _private: (),
}

#[must_use]
pub fn cpu_span() -> CpuSpan {
    #[cfg(feature = "cpu_profiler")]
    {
        google_cpu_profiler::start("./out.profile".as_ref())
    }

    #[cfg(not(feature = "cpu_profiler"))]
    {
        eprintln!(
            r#"cpu profiling is disabled, uncomment `default = [ "cpu_profiler" ]` in Cargo.toml to enable."#
        );
    }

    CpuSpan { _private: () }
}

impl Drop for CpuSpan {
    fn drop(&mut self) {
        #[cfg(feature = "cpu_profiler")]
        {
            google_cpu_profiler::stop();
            let profile_data = std::env::current_dir().unwrap().join("out.profile");
            eprintln!("Profile data saved to:\n\n    {}\n", profile_data.display());
            let mut cmd = std::process::Command::new("pprof");
            cmd.arg("-svg").arg(std::env::current_exe().unwrap()).arg(&profile_data);
            let out = cmd.output();

            match out {
                Ok(out) if out.status.success() => {
                    let svg = profile_data.with_extension("svg");
                    std::fs::write(&svg, out.stdout).unwrap();
                    eprintln!("Profile rendered to:\n\n    {}\n", svg.display());
                }
                _ => {
                    eprintln!("Failed to run:\n\n   {cmd:?}\n");
                }
            }
        }
    }
}

pub fn memory_usage() -> MemoryUsage {
    MemoryUsage::now()
}
