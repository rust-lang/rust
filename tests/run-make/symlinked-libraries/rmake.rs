// Avoid erroring on symlinks pointing to the same file that are present in the library search path.
//
// See <https://github.com/rust-lang/rust/issues/12459>.

//@ ignore-cross-compile
//@ needs-symlink

use run_make_support::{cwd, dynamic_lib_name, path, rfs, rustc};

fn main() {
    rustc().input("foo.rs").arg("-Cprefer-dynamic").run();
    rfs::create_dir_all("other");
    rfs::symlink_file(dynamic_lib_name("foo"), path("other").join(dynamic_lib_name("foo")));
    rustc().input("bar.rs").library_search_path(cwd()).library_search_path("other").run();
}
