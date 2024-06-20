use run_make_support::rustc;
use std::path::Path;

fn main() {
    rustc().input("bar.rs").crate_name("foo").run();
    assert!(Path::new("libfoo.rlib").is_file());
}
