// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Z extra-debug-info
// debugger:break zzz
// debugger:run

// debugger:finish
// debugger:print *t0
// check:$1 = 1
// debugger:print *t1
// check:$2 = 2.5
// debugger:print ret
// check:$3 = {{1, 2.5}, {2.5, 1}}
// debugger:continue

// debugger:finish
// debugger:print *t0
// check:$4 = 3.5
// debugger:print *t1
// check:$5 = 4
// debugger:print ret
// check:$6 = {{3.5, 4}, {4, 3.5}}
// debugger:continue

// debugger:finish
// debugger:print *t0
// check:$7 = 5
// debugger:print *t1
// check:$8 = {a = 6, b = 7.5}
// debugger:print ret
// check:$9 = {{5, {a = 6, b = 7.5}}, {{a = 6, b = 7.5}, 5}}
// debugger:continue

#[deriving(Clone)]
struct Struct {
    a: int,
    b: float
}

fn dup_tup<T0: Clone, T1: Clone>(t0: &T0, t1: &T1) -> ((T0, T1), (T1, T0)) {
    let ret = ((t0.clone(), t1.clone()), (t1.clone(), t0.clone()));
    zzz();
    ret
}

fn main() {

    let _ = dup_tup(&1, &2.5);
    let _ = dup_tup(&3.5, &4_u16);
    let _ = dup_tup(&5, &Struct { a: 6, b: 7.5 });
}

fn zzz() {()}
