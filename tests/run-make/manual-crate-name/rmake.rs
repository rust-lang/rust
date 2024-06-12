use std::path::Path;

use run_make_support::rustc;

fn main() {
    rustc().input("bar.rs").crate_name("foo").run();
    assert!(Path::new("libfoo.rlib").is_file());
}
