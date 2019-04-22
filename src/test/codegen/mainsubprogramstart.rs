// ignore-windows
// ignore-macos

// compile-flags: -g -C no-prepopulate-passes

#![feature(start)]

// CHECK-LABEL: @main
// CHECK: {{.*}}DISubprogram{{.*}}name: "start",{{.*}}DIFlagMainSubprogram{{.*}}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    return 0;
}
