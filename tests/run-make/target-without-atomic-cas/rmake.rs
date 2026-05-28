// ARM Cortex-M are a class of processors supported by the rust compiler. However, they cannot
// support any atomic features, such as Arc. This test simply prints the configuration details of
// one Cortex target, and checks that the compiler does not falsely list atomic support.
// See <https://github.com/rust-lang/rust/pull/36874>.

// ignore-tidy-linelength
//@ needs-llvm-components: arm
// Note: without the needs-llvm-components it will fail on LLVM built without all of the components
// listed above. If any new targets are added, please double-check their respective llvm components
// are specified above.

use run_make_support::rustc;

// The target used below doesn't support atomic CAS operations. Verify that's the case
fn main() {
    rustc()
        .print("cfg")
        .target("thumbv6m-none-eabi")
        .run()
        .assert_stdout_not_contains(r#"target_has_atomic="ptr""#);
}
