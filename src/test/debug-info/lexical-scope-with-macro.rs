// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-android: FIXME(#10381)

// compile-flags:-Z extra-debug-info
// debugger:rbreak zzz
// debugger:run

// debugger:finish
// debugger:print a
// check:$1 = 10
// debugger:print b
// check:$2 = 34
// debugger:continue

// debugger:finish
// debugger:print a
// check:$3 = 890242
// debugger:print b
// check:$4 = 34
// debugger:continue

// debugger:finish
// debugger:print a
// check:$5 = 10
// debugger:print b
// check:$6 = 34
// debugger:continue

// debugger:finish
// debugger:print a
// check:$7 = 102
// debugger:print b
// check:$8 = 34
// debugger:continue

// debugger:finish
// debugger:print a
// check:$9 = 110
// debugger:print b
// check:$10 = 34
// debugger:continue

// debugger:finish
// debugger:print a
// check:$11 = 10
// debugger:print b
// check:$12 = 34
// debugger:continue

// debugger:finish
// debugger:print a
// check:$13 = 10
// debugger:print b
// check:$14 = 34
// debugger:print c
// check:$15 = 400
// debugger:continue

#[feature(macro_rules)];

macro_rules! trivial(
    ($e1:expr) => ($e1)
)

macro_rules! no_new_scope(
    ($e1:expr) => (($e1 + 2) - 1)
)

macro_rules! new_scope(
    () => ({
        let a = 890242;
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

    let a = trivial!(10);
    let b = no_new_scope!(33);

    zzz();
    sentinel();

    new_scope!();

    zzz();
    sentinel();

    shadow_within_macro!(100);

    zzz();
    sentinel();

    let c = dup_expr!(10 * 20);

    zzz();
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
