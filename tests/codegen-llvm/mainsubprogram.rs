// This test depends on a patch that was committed to upstream LLVM
// before 4.0, formerly backported to the Rust LLVM fork.

//@ ignore-apple
//@ ignore-wasi

//@ compile-flags: -g -C no-prepopulate-passes

// CHECK-LABEL: @main
// CHECK: {{.*}}DISubprogram{{.*}}name: "main",{{.*}}DI{{(SP)?}}FlagMainSubprogram{{.*}}

pub fn main() {}
