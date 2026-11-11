// Defining a crate that provides panic handling as an external crate
// could uselessly trigger the "unused external crate" lint. In this test,
// if the lint is triggered, it will trip #![deny(unused_extern_crates)],
// and cause the test to fail.
// See https://github.com/rust-lang/rust/issues/53964

use run_make_support::rustc;

fn main() {
    rustc().input("panic.rs").run();
    rustc().input("app.rs").panic("abort").emit("obj").run();
}
