// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(naked_functions)]

// CHECK: Function Attrs: naked uwtable
// CHECK-NEXT: define internal void @naked_empty()
#[no_mangle]
#[naked]
fn naked_empty() {
    // CHECK: ret void
}

// CHECK: Function Attrs: naked uwtable
#[no_mangle]
#[naked]
// CHECK-NEXT: define internal void @naked_with_args(i{{[0-9]+}})
fn naked_with_args(a: isize) {
    // CHECK: %a = alloca i{{[0-9]+}}
    // CHECK: ret void
    &a; // keep variable in an alloca
}

// CHECK: Function Attrs: naked uwtable
// CHECK-NEXT: define internal i{{[0-9]+}} @naked_with_return()
#[no_mangle]
#[naked]
fn naked_with_return() -> isize {
    // CHECK: ret i{{[0-9]+}} 0
    0
}

// CHECK: Function Attrs: naked uwtable
// CHECK-NEXT: define internal i{{[0-9]+}} @naked_with_args_and_return(i{{[0-9]+}})
#[no_mangle]
#[naked]
fn naked_with_args_and_return(a: isize) -> isize {
    // CHECK: %a = alloca i{{[0-9]+}}
    // CHECK: ret i{{[0-9]+}} %{{[0-9]+}}
    &a; // keep variable in an alloca
    a
}

// CHECK: Function Attrs: naked uwtable
// CHECK-NEXT: define internal void @naked_recursive()
#[no_mangle]
#[naked]
fn naked_recursive() {
    // CHECK: call void @naked_empty()
    naked_empty();
    // CHECK: %{{[0-9]+}} = call i{{[0-9]+}} @naked_with_return()
    naked_with_args(
        // CHECK: %{{[0-9]+}} = call i{{[0-9]+}} @naked_with_args_and_return(i{{[0-9]+}} %{{[0-9]+}})
        naked_with_args_and_return(
            // CHECK: call void @naked_with_args(i{{[0-9]+}} %{{[0-9]+}})
            naked_with_return()
        )
    );
}
