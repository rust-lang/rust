// This test depends on a patch that was committed to upstream LLVM
// after 5.0, then backported to the Rust LLVM fork.

// ignore-windows
// ignore-macos

// compile-flags: -g -C no-prepopulate-passes

// CHECK-LABEL: @main
// CHECK: {{.*}}DICompositeType{{.*}}name: "vtable",{{.*}}vtableHolder:{{.*}}

pub trait T {
}

impl T for f64 {
}

pub fn main() {
    let d = 23.0f64;
    let td = &d as &T;
}
