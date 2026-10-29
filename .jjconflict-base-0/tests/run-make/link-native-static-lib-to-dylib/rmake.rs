// Regression test for <https://github.com/rust-lang/rust/issues/15460>.

//@ ignore-cross-compile

use run_make_support::{build_native_static_lib, run, rustc};

fn main() {
    build_native_static_lib("foo");

    rustc().input("foo.rs").extra_filename("-383hf8").arg("-Cprefer-dynamic").run();
    rustc().input("bar.rs").run();

    run("bar");
}
