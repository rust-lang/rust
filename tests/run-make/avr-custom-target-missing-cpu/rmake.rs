// Custom AVR targets can reach ELF e_flags emission without an explicit CPU
// Make sure that reports the normal missing-CPU diagnostic instead of ICEing
//
//@ needs-llvm-components: avr

use run_make_support::rustc;

fn main() {
    rustc()
        .arg("-Zunstable-options")
        .input("foo.rs")
        .target("avr-custom-missing-cpu.json")
        .crate_type("lib")
        .run_fail()
        .assert_stderr_contains("target requires explicitly specifying a cpu with `-C target-cpu`");
}
