//@ compile-flags: -Cno-prepopulate-passes
//@ needs-unwind
#![crate_type = "lib"]

// This test checks that drop calls in unwind landing pads
// get the `cold` attribute.

// CHECK-LABEL: @check_cold
// CHECK: {{(call|invoke) void .+}}drop_glue{{.+}} [[ATTRIBUTES:#[0-9]+]]
// CHECK: attributes [[ATTRIBUTES]] = { cold }

// A drop that provably cannot unwind is called with a plain `call`, so the glue here is one that
// does unwind.

struct Panicky;

impl Drop for Panicky {
    fn drop(&mut self) {
        panic!("drop");
    }
}

#[no_mangle]
pub fn check_cold(f: fn(), x: Box<Panicky>) {
    // this may unwind
    f();
}
