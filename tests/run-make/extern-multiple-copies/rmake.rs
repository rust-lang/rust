//@ needs-target-std
//
// In this test, the rust library foo1 exists in two different locations, but only one
// is required by the --extern flag. This test checks that the copy is ignored (as --extern
// demands fetching only the original instance of foo1) and that no error is emitted, resulting
// in successful compilation.
// https://github.com/rust-lang/rust/pull/29961

use run_make_support::{path, rfs, rust_lib_name, rustc};

fn main() {
    rustc().input("foo1.rs").run();
    rustc().input("foo2.rs").run();
    rfs::create_dir("foo");
    rfs::copy(rust_lib_name("foo1"), path("foo").join(rust_lib_name("foo1")));
    rustc().input("bar.rs").extern_("foo1", rust_lib_name("foo1")).library_search_path("foo").run();
}
