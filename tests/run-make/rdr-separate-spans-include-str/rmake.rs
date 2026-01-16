// Verify ExpnData spans decoded from metadata are usable for include_str!
// when building with -Z separate-spans.
//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{run, rust_lib_name, rustc};

fn main() {
    rustc().input("dep.rs").crate_name("dep").crate_type("rlib").arg("-Zseparate-spans").run();

    rustc()
        .input("main.rs")
        .arg("-Zseparate-spans")
        .arg("--extern")
        .arg(format!("dep={}", rust_lib_name("dep")))
        .run();

    run("main");
}
