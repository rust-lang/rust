// aux-build:test-macros.rs
// compile-flags: -Z span-debug
// revisions: local remapped
// [remapped]compile-flags: --remap-path-prefix={{src-base}}=remapped

// The remapped paths are not normalized by compiletest.
// normalize-stdout-test: "\\(proc-macro|pretty-print-hack)" -> "/$1"
// normalize-stderr-test: "\\(proc-macro|pretty-print-hack)" -> "/$1"

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use] extern crate test_macros;

mod first {
    include!("pretty-print-hack/allsorts-rental-0.5.6/src/lib.rs");
}

mod second {
    include!("pretty-print-hack/rental-0.5.5/src/lib.rs");
}

fn main() {}
