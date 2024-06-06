use run_make_support::{rustc, tmp_dir};

fn main() {
    rustc().input("bar.rs").crate_name("foo").run();
    assert!(tmp_dir().join("libfoo.rlib").is_file());
}
