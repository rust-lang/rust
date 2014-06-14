// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

#![feature(managed_boxes)]

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run
// gdb-command:finish

// gdb-command:print *stack_val_ref
// gdb-check:$1 = {-14, -19}

// gdb-command:print *ref_to_unnamed
// gdb-check:$2 = {-15, -20}

// gdb-command:print *managed_val_ref
// gdb-check:$3 = {-16, -21}

// gdb-command:print *unique_val_ref
// gdb-check:$4 = {-17, -22}

#![allow(unused_variable)]

use std::gc::{Gc, GC};

fn main() {
    let stack_val: (i16, f32) = (-14, -19f32);
    let stack_val_ref: &(i16, f32) = &stack_val;
    let ref_to_unnamed: &(i16, f32) = &(-15, -20f32);

    let managed_val: Gc<(i16, f32)> = box(GC) (-16, -21f32);
    let managed_val_ref: &(i16, f32) = managed_val;

    let unique_val: Box<(i16, f32)> = box() (-17, -22f32);
    let unique_val_ref: &(i16, f32) = unique_val;

    zzz();
}

fn zzz() {()}
