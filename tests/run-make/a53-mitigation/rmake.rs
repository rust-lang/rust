//@ only-aarch64-unknown-linux-gnu
//@ needs-target-std

use run_make_support::rustc;

fn main() {
    rustc()
        .input("empty.rs")
        .print("link-args")
        .run_unchecked()
        .assert_stdout_contains("-mfix-cortex-a53-843419");
}
