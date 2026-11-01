//@ needs-target-std
use run_make_support::{path, rustc};

fn main() {
    rustc().input("bar.rs").crate_name("foo").run();
    assert!(path("libfoo.rlib").is_file());
}
