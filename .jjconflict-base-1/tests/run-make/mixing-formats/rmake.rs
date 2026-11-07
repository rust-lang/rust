// Testing various mixings of rlibs and dylibs. Makes sure that it's possible to
// link an rlib to a dylib. The dependency tree among the file looks like:
//
//                 foo
//               /     \
//             bar1   bar2
//             /    \ /
//          baz    baz2
//
// This is generally testing the permutations of the foo/bar1/bar2 layer against
// the baz/baz2 layer

//@ ignore-cross-compile

use run_make_support::{run_in_tmpdir, rustc};

fn main() {
    run_in_tmpdir(|| {
        // Building just baz
        rustc().crate_type("rlib").input("foo.rs").run();
        rustc().crate_type("dylib").input("bar1.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("dylib,rlib").input("baz.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("bin").input("baz.rs").run();
    });
    run_in_tmpdir(|| {
        rustc().crate_type("dylib").input("foo.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("rlib").input("bar1.rs").run();
        rustc().crate_type("dylib,rlib").input("baz.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("bin").input("baz.rs").run();
    });
    run_in_tmpdir(|| {
        // Building baz2
        rustc().crate_type("rlib").input("foo.rs").run();
        rustc().crate_type("dylib").input("bar1.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("dylib").input("bar2.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("dylib").input("baz2.rs").run_fail().assert_exit_code(1);
        rustc().crate_type("bin").input("baz2.rs").run_fail().assert_exit_code(1);
    });
    run_in_tmpdir(|| {
        rustc().crate_type("rlib").input("foo.rs").run();
        rustc().crate_type("rlib").input("bar1.rs").run();
        rustc().crate_type("dylib").input("bar2.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("dylib,rlib").input("baz2.rs").run();
        rustc().crate_type("bin").input("baz2.rs").run();
    });
    run_in_tmpdir(|| {
        rustc().crate_type("rlib").input("foo.rs").run();
        rustc().crate_type("dylib").input("bar1.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("rlib").input("bar2.rs").run();
        rustc().crate_type("dylib,rlib").input("baz2.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("bin").input("baz2.rs").run();
    });
    run_in_tmpdir(|| {
        rustc().crate_type("rlib").input("foo.rs").run();
        rustc().crate_type("rlib").input("bar1.rs").run();
        rustc().crate_type("rlib").input("bar2.rs").run();
        rustc().crate_type("dylib,rlib").input("baz2.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("bin").input("baz2.rs").run();
    });
    run_in_tmpdir(|| {
        rustc().crate_type("dylib").input("foo.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("rlib").input("bar1.rs").run();
        rustc().crate_type("rlib").input("bar2.rs").run();
        rustc().crate_type("dylib,rlib").input("baz2.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("bin").input("baz2.rs").run();
    });
    run_in_tmpdir(|| {
        rustc().crate_type("dylib").input("foo.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("dylib").input("bar1.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("rlib").input("bar2.rs").run();
        rustc().crate_type("dylib,rlib").input("baz2.rs").run();
        rustc().crate_type("bin").input("baz2.rs").run();
    });
    run_in_tmpdir(|| {
        rustc().crate_type("dylib").input("foo.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("rlib").input("bar1.rs").run();
        rustc().crate_type("dylib").input("bar2.rs").arg("-Cprefer-dynamic").run();
        rustc().crate_type("dylib,rlib").input("baz2.rs").run();
        rustc().crate_type("bin").input("baz2.rs").run();
    });
    rustc().crate_type("dylib").input("foo.rs").arg("-Cprefer-dynamic").run();
    rustc().crate_type("dylib").input("bar1.rs").arg("-Cprefer-dynamic").run();
    rustc().crate_type("dylib").input("bar2.rs").arg("-Cprefer-dynamic").run();
    rustc().crate_type("dylib,rlib").input("baz2.rs").run();
    rustc().crate_type("bin").input("baz2.rs").run();
}
