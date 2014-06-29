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

// compile-flags:-g
// gdb-command:rbreak zzz
// gdb-command:run

// gdb-command:finish
// gdb-command:print a
// gdb-check:$1 = 10
// gdb-command:print b
// gdb-check:$2 = 34
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$3 = 890242
// gdb-command:print b
// gdb-check:$4 = 34
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$5 = 10
// gdb-command:print b
// gdb-check:$6 = 34
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$7 = 102
// gdb-command:print b
// gdb-check:$8 = 34
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$9 = 110
// gdb-command:print b
// gdb-check:$10 = 34
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$11 = 10
// gdb-command:print b
// gdb-check:$12 = 34
// gdb-command:continue

// gdb-command:finish
// gdb-command:print a
// gdb-check:$13 = 10
// gdb-command:print b
// gdb-check:$14 = 34
// gdb-command:print c
// gdb-check:$15 = 400
// gdb-command:continue

#![feature(macro_rules)]

macro_rules! trivial(
    ($e1:expr) => ($e1)
)

macro_rules! no_new_scope(
    ($e1:expr) => (($e1 + 2) - 1)
)

macro_rules! new_scope(
    () => ({
        let a = 890242i;
        zzz();
        sentinel();
    })
)

macro_rules! shadow_within_macro(
    ($e1:expr) => ({
        let a = $e1 + 2;

        zzz();
        sentinel();

        let a = $e1 + 10;

        zzz();
        sentinel();
    })
)


macro_rules! dup_expr(
    ($e1:expr) => (($e1) + ($e1))
)


fn main() {

    let a = trivial!(10i);
    let b = no_new_scope!(33i);

    zzz();
    sentinel();

    new_scope!();

    zzz();
    sentinel();

    shadow_within_macro!(100i);

    zzz();
    sentinel();

    let c = dup_expr!(10i * 20);

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
