// This test verifies that functions including a leading question mark
// in their export_name attribute successfully compile.
// This is only an issue on Windows 32-bit.

use run_make_support::{run, run_fail, rustc};

fn main() {
    rustc().input("main.rs").target("i686-pc-windows-msvc").run();
}
