// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=eager

#![deny(dead_code)]

trait Trait : Sized {
    fn foo(self) -> Self { self }
}

impl Trait for u32 {
    fn foo(self) -> u32 { self }
}

impl Trait for char {
}

fn take_foo_once<T, F: FnOnce(T) -> T>(f: F, arg: T) -> T {
    (f)(arg)
}

fn take_foo<T, F: Fn(T) -> T>(f: F, arg: T) -> T {
    (f)(arg)
}

fn take_foo_mut<T, F: FnMut(T) -> T>(mut f: F, arg: T) -> T {
    (f)(arg)
}

//~ TRANS_ITEM fn trait_method_as_argument::main[0]
fn main() {
    //~ TRANS_ITEM fn trait_method_as_argument::take_foo_once[0]<u32, fn(u32) -> u32>
    //~ TRANS_ITEM fn trait_method_as_argument::{{impl}}[0]::foo[0]
    take_foo_once(Trait::foo, 0u32);

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo_once[0]<char, fn(char) -> char>
    //~ TRANS_ITEM fn trait_method_as_argument::Trait[0]::foo[0]<char>
    take_foo_once(Trait::foo, 'c');

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo[0]<u32, fn(u32) -> u32>
    take_foo(Trait::foo, 0u32);

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo[0]<char, fn(char) -> char>
    take_foo(Trait::foo, 'c');

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo_mut[0]<u32, fn(u32) -> u32>
    take_foo_mut(Trait::foo, 0u32);

    //~ TRANS_ITEM fn trait_method_as_argument::take_foo_mut[0]<char, fn(char) -> char>
    take_foo_mut(Trait::foo, 'c');
}

//~ TRANS_ITEM drop-glue i8
