//@ only-windows
// Reason: dos devices are a Windows thing

use run_make_support::{path, rustc, static_lib_name};

fn main() {
    rustc().input(r"\\.\NUL").crate_type("staticlib").run();
    rustc().input(r"\\?\NUL").crate_type("staticlib").run();

    assert!(path(&static_lib_name("rust_out")).exists());
}
