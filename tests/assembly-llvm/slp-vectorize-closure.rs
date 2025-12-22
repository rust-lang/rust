//! Loop inside closure correctly compiled to assembly using SSE instructions.
//! Regression test for <https://github.com/rust-lang/rust/issues/120189>.
//! also see <https://godbolt.org/z/W1Yc4s3xo>
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//@ only-x86_64
// to test SSE instructions

#![crate_type = "lib"]

// CHECK-LABEL: for_in_closure
// CHECK: {{paddb|psubb}}
#[inline(never)]
#[no_mangle]
pub fn for_in_closure() {
    let mut v = [[0u8; 4]; 60];

    let mut closure = || {
        for item in &mut v {
            item[0] += 1;
            item[1] += 1;
            item[2] += 1;
            item[3] += 1;
        }
    };

    closure();
}
