//@ compile-flags: -O
//@ ignore-wasm FIXME: LLVM does not vectorize loops on WASM so this doesn't get optimized
#![crate_type = "lib"]

// Test that simple iterator-based loops of length > 101 are fully optimized away.
// See <https://github.com/rust-lang/rust/issues/112169>

// CHECK-LABEL: issue_112169()
#[no_mangle]
pub fn issue_112169() -> i32 {
    // CHECK-NEXT: {{.*}}:
    // CHECK-NEXT: ret i32 102
    let mut s = 0;

    for i in 0..102 {
        if i == 0 {
            s = i;
        }

        s += 1;
    }

    s
}
