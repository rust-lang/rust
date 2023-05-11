// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

pub trait T {}

// CHECK-LABEL: @copy_fat_ptr
#[no_mangle]
pub fn copy_fat_ptr(x: &T) {
// CHECK-NOT: extractvalue
    let x2 = x;
}
