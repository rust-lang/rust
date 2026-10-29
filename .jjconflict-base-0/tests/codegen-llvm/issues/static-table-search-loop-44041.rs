// Tests that a loop searching a small static array is unrolled and simplified
// down to a single comparison, with none of the unrolled branches left behind.
// See <https://github.com/rust-lang/rust/issues/44041>.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

static TABLE: [i32; 4] = [0; 4];

// CHECK-LABEL: @exists_in_table(
// CHECK-NOT: br {{.*}}
// CHECK: icmp eq i32
// CHECK-NOT: br {{.*}}
// CHECK: ret i1
#[no_mangle]
pub fn exists_in_table(v: i32) -> bool {
    for &x in TABLE.iter() {
        if x == v {
            return true;
        }
    }
    false
}
