#![crate_name = "interesting"]
#![crate_type = "rlib"]

extern crate opaque;

#[no_mangle]
#[inline(never)]
pub fn function_called_once() {
    opaque::foo();
}

// CHECK-LABEL: @function_called_once
// CHECK-SAME: !prof [[function_called_once_id:![0-9]+]] {
// CHECK: "CG Profile"
// CHECK-NOT: "CG Profile"
// CHECK-DAG: [[function_called_once_id]] = !{!"function_entry_count", i64 1}
