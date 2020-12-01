// This test is for *-windows-msvc only.
// ignore-android
// ignore-dragonfly
// ignore-emscripten
// ignore-freebsd
// ignore-haiku
// ignore-ios
// ignore-linux
// ignore-macos
// ignore-netbsd
// ignore-openbsd
// ignore-solaris
// ignore-sgx

// compile-flags: -C no-prepopulate-passes -C panic=abort -O

#![crate_type = "lib"]

// CHECK: Function Attrs: nounwind uwtable
// CHECK-NEXT: define void @normal_uwtable()
#[no_mangle]
pub fn normal_uwtable() {
}

// CHECK: Function Attrs: nounwind uwtable
// CHECK-NEXT: define void @extern_uwtable()
#[no_mangle]
pub extern fn extern_uwtable() {
}
