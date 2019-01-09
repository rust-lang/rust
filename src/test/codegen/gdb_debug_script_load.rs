// ignore-tidy-linelength
// ignore-windows
// ignore-macos

// compile-flags: -g -C no-prepopulate-passes

#![feature(start)]

// CHECK-LABEL: @main
// CHECK: load volatile i8, i8* getelementptr inbounds ([[B:\[[0-9]* x i8\]]], [[B]]* @__rustc_debug_gdb_scripts_section__, i32 0, i32 0), align 1

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    return 0;
}
