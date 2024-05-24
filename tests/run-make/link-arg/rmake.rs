// In 2016, the rustc flag "-C link-arg" was introduced - it can be repeatedly used
// to add single arguments to the linker. This test passes 2 arguments to the linker using it,
// then checks that the compiler's output contains the arguments passed to it.
// This ensures that the compiler successfully parses this flag.
// See https://github.com/rust-lang/rust/pull/36574

use run_make_support::rustc;

fn main() {
    let output = String::from_utf8(
        rustc()
            .input("empty.rs")
            .link_arg("-lfoo")
            .link_arg("-lbar")
            .print("link-args")
            .command_output()
            .stdout,
    )
    .unwrap();
    assert!(
        output.contains("lfoo") || output.contains("lbar"),
        "The output did not contain the expected \"lfoo\" or \"lbar\" strings."
    );
}
