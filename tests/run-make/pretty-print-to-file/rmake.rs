// The "pretty-printer" of rustc translates source code into other formats,
// which is useful for debugging. This test checks the "normal" version of
// -Zunpretty, which should format the poorly formatted input.rs into a one-line
// function identical to the one in input.pp.
// See https://github.com/rust-lang/rust/commit/da25539c1ab295ec40261109557dd4526923928c

use run_make_support::{diff, rustc};

fn main() {
    rustc().output("input.out").arg("-Zunpretty=normal").input("input.rs").run();
    diff().expected_file("input.out").actual_file("input.pp").run();
}
