//@ needs-target-std
//
// Regression test for <https://github.com/rust-lang/rust/issues/159677>.
//
// Ensures two builds of `client.rs` produce identical metadata
// even if there is an unrelated crate on the search path.

use run_make_support::{rfs, rustc};

fn main() {
    rustc().input("foo.rs").crate_type("rlib").run();
    rustc()
        .input("client.rs")
        .crate_type("rlib")
        .emit("metadata")
        .library_search_path(".")
        .output("client1.rmeta")
        .run();

    rustc().input("foo_bar.rs").crate_type("rlib").run();
    rustc()
        .input("client.rs")
        .crate_type("rlib")
        .emit("metadata")
        .library_search_path(".")
        .output("client2.rmeta")
        .run();

    assert_eq!(rfs::read("client1.rmeta"), rfs::read("client2.rmeta"));
}
