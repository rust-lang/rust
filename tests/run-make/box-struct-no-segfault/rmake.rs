// The crate "foo" tied to this test executes a very specific function,
// which involves boxing an instance of the struct Foo. However,
// this once caused a segmentation fault in cargo release builds due to an LLVM
// incorrect assertion.
// This test checks that this bug does not resurface.
// See https://github.com/rust-lang/rust/issues/28766

use run_make_support::{rustc, tmp_dir};

fn main() {
    rustc().opt().input("foo.rs").run();
    rustc().opt().library_search_path(tmp_dir()).input("main.rs").run();
}
