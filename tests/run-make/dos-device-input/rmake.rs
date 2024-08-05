//@ only-windows
// Reason: dos devices are a Windows thing

use run_make_support::rustc;

fn main() {
    rustc().input(r"\\.\NUL").crate_type("staticlib").run();
    rustc().input(r"\\?\NUL").crate_type("staticlib").run();
}
