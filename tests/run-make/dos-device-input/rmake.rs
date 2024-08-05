//@ only-windows
// Reason: dos devices are a Windows thing

use std::path::Path;

use run_make_support::{rustc, static_lib_name};

fn main() {
    rustc().input(r"\\.\NUL").crate_type("staticlib").run();
    rustc().input(r"\\?\NUL").crate_type("staticlib").run();

    assert!(Path::new(&static_lib_name("rust_out")).exists());
}
