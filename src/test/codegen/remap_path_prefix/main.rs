// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows
// ignore-tidy-linelength

// compile-flags: -g  -C no-prepopulate-passes -Zremap-path-prefix-from={{cwd}} -Zremap-path-prefix-to=/the/cwd -Zremap-path-prefix-from={{src-base}} -Zremap-path-prefix-to=/the/src
// aux-build:remap_path_prefix_aux.rs

extern crate remap_path_prefix_aux;

// Here we check that submodules and include files are found using the path without
// remapping. This test requires that rustc is called with an absolute path.
mod aux_mod;
include!("aux_mod.rs");

// Here we check that the expansion of the file!() macro is mapped.
// CHECK: internal constant [34 x i8] c"/the/src/remap_path_prefix/main.rs"
pub static FILE_PATH: &'static str = file!();

fn main() {
    remap_path_prefix_aux::some_aux_function();
    aux_mod::some_aux_mod_function();
    some_aux_mod_function();
}

// Here we check that local debuginfo is mapped correctly.
// CHECK: !DIFile(filename: "/the/src/remap_path_prefix/main.rs", directory: "/the/cwd/")

// And here that debuginfo from other crates are expanded to absolute paths.
// CHECK: !DIFile(filename: "/the/aux-src/remap_path_prefix_aux.rs", directory: "")
