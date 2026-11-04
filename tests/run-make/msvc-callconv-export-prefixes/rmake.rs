// Regression test for #165299.
// Special-prefix vectorcall exports must not cause unresolved linker symbols.
//
// Unpatched rustc fails with LNK2001 for ctl_vectorcall@@16
// and ?q_vectorcall@@16.

//@ only-x86_64-pc-windows-msvc
//@ needs-crate-type: cdylib

use run_make_support::rustc;

fn main() {
    rustc()
        .input("exports.rs")
        .crate_type("cdylib")
        .arg("-Cpanic=abort")
        .arg("-Clink-arg=/NOENTRY")
        .run();
}
