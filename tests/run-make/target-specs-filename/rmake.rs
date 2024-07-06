// This test checks the rustc can accept target-spec-json with cases
// including file extension - it could be in upper and lower case.
// Used to test: print target-spec-json and cfg
// Issue: https://github.com/rust-lang/rust/issues/127387

use run_make_support::rustc;

fn main() {
    // This is a matrix [files x args], files for target, args for print.
    for file in [
        "normal-target-triple",
        "normal-target-triple.json",
        "ext-in-caps.JSON",
        "ALL-IN-CAPS.JSON",
    ] {
        for args in [["cfg"].as_slice(), &["target-spec-json", "-Zunstable-options"]] {
            let output = rustc().arg("--print").args(args).args(["--target", file]).run();
            // check that the target triple is read correctly
            output.assert_stdout_contains("x86_64");
        }
    }
}
