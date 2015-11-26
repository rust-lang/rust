// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks for private types in public interfaces

struct Priv;

pub use self::private::public;

mod private {
    pub type Priv = super::Priv; //~ WARN private type in public interface

    pub fn public(_x: Priv) {
    }
}

struct __CFArray;
pub type CFArrayRef = *const __CFArray; //~ WARN private type in public interface
trait Pointer { type Pointee; }
impl<T> Pointer for *const T { type Pointee = T; }
pub type __CFArrayRevealed = <CFArrayRef as Pointer>::Pointee;
//~^ WARN private type in public interface

type Foo = u8;
pub fn foo(f: Foo) {} //~ ERROR private type in public interface

pub trait Exporter {
    type Output;
}
pub struct Helper;

pub fn block() -> <Helper as Exporter>::Output {
    struct Inner;
    impl Inner {
        fn poke(&self) { println!("Hello!"); }
    }

    impl Exporter for Helper {
        type Output = Inner; //~ WARN private type in public interface
    }

    Inner
}

fn main() {
    block().poke();
}
