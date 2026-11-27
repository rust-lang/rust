// When an empty output file is passed to rustc, the ensuing error message
// should be clear. However, calling file_stem on an empty path returns None,
// which, when unwrapped, causes a panic, stopping execution of rustc
// and printing an obscure message instead of reaching the helpful
// error message. This test checks that the panic does not occur.
// See https://github.com/rust-lang/rust/pull/26199

use run_make_support::rustc;

fn main() {
    let output = rustc().output("").stdin_buf(b"fn main() {}").run_fail();
    output.assert_stderr_not_contains("panic");
}
