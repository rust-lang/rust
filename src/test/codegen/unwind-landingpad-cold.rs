// no-system-llvm: needs #92110
// compile-flags: -Cno-prepopulate-passes
#![crate_type = "lib"]

// This test checks that drop calls in unwind landing pads
// get the `cold` attribute.

// CHECK-LABEL: @check_cold
// CHECK: call void {{.+}}drop_in_place{{.+}} [[ATTRIBUTES:#[0-9]+]]
// CHECK: attributes [[ATTRIBUTES]] = { cold }
#[no_mangle]
pub fn check_cold(f: fn(), x: Box<u32>) {
    // this may unwind
    f();
}
