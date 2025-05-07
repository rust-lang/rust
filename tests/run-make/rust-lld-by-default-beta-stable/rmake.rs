// Ensure that rust-lld is *not* used as the default linker on `x86_64-unknown-linux-gnu` on stable
// or beta.

//@ ignore-nightly
//@ only-x86_64-unknown-linux-gnu

use run_make_support::linker::assert_rustc_doesnt_use_lld;
use run_make_support::rustc;

fn main() {
    // A regular compilation should not use rust-lld by default. We'll check that by asking the
    // linker to display its version number with a link-arg.
    assert_rustc_doesnt_use_lld(rustc().input("main.rs"));
}
