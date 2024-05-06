//@ ignore-windows
//@ ignore-macos
//@ ignore-wasi wasi codegens the main symbol differently

//@ compile-flags: -g -C no-prepopulate-passes

#![feature(start)]

// CHECK-LABEL: @main
// CHECK: {{.*}}DISubprogram{{.*}}name: "start",{{.*}}DI{{(SP)?}}FlagMainSubprogram{{.*}}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    return 0;
}
