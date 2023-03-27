// ignore-windows
//

// compile-flags: -g  -C no-prepopulate-passes --remap-path-prefix={{cwd}}=/the/cwd --remap-path-prefix={{src-base}}=/the/src -Zinline-mir=no
// aux-build:remap_path_prefix_aux.rs

extern crate remap_path_prefix_aux;

// Here we check that submodules and include files are found using the path without
// remapping. This test requires that rustc is called with an absolute path.
mod aux_mod;
include!("aux_mod.rs");

// Here we check that the expansion of the file!() macro is mapped.
// CHECK: @alloc_af9d0c7bc6ca3c31bb051d2862714536 = private unnamed_addr constant <{ [34 x i8] }> <{ [34 x i8] c"/the/src/remap_path_prefix/main.rs" }>
pub static FILE_PATH: &'static str = file!();

fn main() {
    remap_path_prefix_aux::some_aux_function();
    aux_mod::some_aux_mod_function();
    some_aux_mod_function();
}

// Here we check that local debuginfo is mapped correctly.
// CHECK: !DIFile(filename: "/the/src/remap_path_prefix/main.rs", directory: ""

// And here that debuginfo from other crates are expanded to absolute paths.
// CHECK: !DIFile(filename: "/the/aux-src/remap_path_prefix_aux.rs", directory: ""
