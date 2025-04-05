// The `crate_name` rustc flag should have higher priority
// over `#![crate_name = "foo"]` defined inside the source code.
// This test has a conflict between crate_names defined in the .rs files
// and the compiler flags, and checks that the flag is favoured each time.
// See https://github.com/rust-lang/rust/pull/15518

//@ ignore-nvptx64 (no target std)

use run_make_support::{bin_name, rfs, rustc, target};

fn main() {
    rustc().target(target()).input("foo.rs").run();
    rfs::remove_file(bin_name("foo"));
    rustc().target(target()).input("foo.rs").crate_name("bar").run();
    rfs::remove_file(bin_name("bar"));
    rustc().target(target()).input("foo1.rs").run();
    rfs::remove_file(bin_name("foo"));
    rustc().target(target()).input("foo1.rs").output(bin_name("bar1")).run();
    rfs::remove_file(bin_name("bar1"));
}
