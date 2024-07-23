//@ ignore-cross-compile

use run_make_support::{cwd, run, rustc};

// Attempt to build this dependency tree:
//
//    A.1   A.2
//     |\    |
//     | \   |
//     B  \  C
//      \ | /
//       \|/
//        D
//
// Note that A.1 and A.2 are crates with the same name.

// original Makefile at https://github.com/rust-lang/rust/commit/cc3c8bbfaf5af19caf3deb131a995a65ca4674f9#diff-23d3f6bcee0c974ae7c793ab6f87c454cffd1f5c9ec17ce86aaf2cf8bc6e5397

fn main() {
    rustc().metadata("1").extra_filename("-1").input("a.rs").run();
    rustc().metadata("2").extra_filename("-2").input("a.rs").run();
    rustc().input("b.rs").extern_("a", "liba-1.rlib").run();
    rustc().input("c.rs").extern_("a", "liba-2.rlib").run();
    println!("before");
    rustc().cfg("before").input("d.rs").extern_("a", "liba-1.rlib").run();
    run("d");
    println!("after");
    rustc().cfg("after").input("d.rs").extern_("a", "liba-1.rlib").run();
    run("d");
}
