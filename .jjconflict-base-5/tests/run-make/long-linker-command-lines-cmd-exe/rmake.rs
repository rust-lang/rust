// Like the `long-linker-command-lines` test this test attempts to blow
// a command line limit for running the linker. Unlike that test, however,
// this test is testing `cmd.exe` specifically rather than the OS.
//
// Unfortunately, the maximum length of the string that you can use at the
// command prompt (`cmd.exe`) is 8191 characters.
// Anyone scripting rustc's linker
// is probably using a `*.bat` script and is likely to hit this limit.
//
// This test uses a `foo.bat` script as the linker which just simply
// delegates back to this program. The compiler should use a lower
// limit for arguments before passing everything via `@`, which
// means that everything should still succeed here.
// See https://github.com/rust-lang/rust/pull/47507

//@ ignore-cross-compile
// Reason: the compiled binary is executed
//@ only-windows
// Reason: this test is specific to Windows executables

use run_make_support::{run, rustc};

fn main() {
    rustc().input("foo.rs").arg("-g").run();
    run("foo");
}
