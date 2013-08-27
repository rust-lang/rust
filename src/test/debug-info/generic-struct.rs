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
// debugger:print int_int
// check:$1 = {key = 0, value = 1}
// debugger:print int_float
// check:$2 = {key = 2, value = 3.5}
// debugger:print float_int
// check:$3 = {key = 4.5, value = 5}
// debugger:print float_int_float
// check:$4 = {key = 6.5, value = {key = 7, value = 8.5}}

struct AGenericStruct<TKey, TValue> {
    key: TKey,
    value: TValue
}

fn main() {

    let int_int = AGenericStruct { key: 0, value: 1 };
    let int_float = AGenericStruct { key: 2, value: 3.5 };
    let float_int = AGenericStruct { key: 4.5, value: 5 };
    let float_int_float = AGenericStruct { key: 6.5, value: AGenericStruct { key: 7, value: 8.5 } };

    zzz();
}

fn zzz() {()}
