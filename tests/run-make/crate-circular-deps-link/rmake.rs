//@ ignore-cross-compile

// Test that previously triggered a linker failure with root cause
// similar to one found in the issue #69368.
//
// The crate that provides oom lang item is missing some other lang
// items. Necessary to prevent the use of start-group / end-group.
//
// The weak lang items are defined in a separate compilation units,
// so that linker could omit them if not used.
//
// The crates that need those weak lang items are dependencies of
// crates that provide them.
// See https://github.com/rust-lang/rust/issues/69371

use run_make_support::rustc;

fn main() {
    rustc().input("a.rs").run();
    rustc().input("b.rs").run();
    rustc().input("c.rs").run();
}
