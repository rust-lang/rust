// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty issue #37201

struct X { val: i32 }
impl std::ops::Deref for X {
    type Target = i32;
    fn deref(&self) -> &i32 { &self.val }
}


trait            M                   { fn m(self); }
impl             M for i32           { fn m(self) { println!("i32::m()"); } }
impl             M for X             { fn m(self) { println!("X::m()"); } }
impl<'a>         M for &'a X         { fn m(self) { println!("&X::m()"); } }
impl<'a, 'b>     M for &'a &'b X     { fn m(self) { println!("&&X::m()"); } }
impl<'a, 'b, 'c> M for &'a &'b &'c X { fn m(self) { println!("&&&X::m()"); } }

trait            RefM                   { fn refm(&self); }
impl             RefM for i32           { fn refm(&self) { println!("i32::refm()"); } }
impl             RefM for X             { fn refm(&self) { println!("X::refm()"); } }
impl<'a>         RefM for &'a X         { fn refm(&self) { println!("&X::refm()"); } }
impl<'a, 'b>     RefM for &'a &'b X     { fn refm(&self) { println!("&&X::refm()"); } }
impl<'a, 'b, 'c> RefM for &'a &'b &'c X { fn refm(&self) { println!("&&&X::refm()"); } }

struct Y { val: i32 }
impl std::ops::Deref for Y {
    type Target = i32;
    fn deref(&self) -> &i32 { &self.val }
}

struct Z { val: Y }
impl std::ops::Deref for Z {
    type Target = Y;
    fn deref(&self) -> &Y { &self.val }
}

struct A;
impl std::marker::Copy for A {}
impl Clone for A { fn clone(&self) -> Self { *self } }
impl             M for             A { fn m(self) { println!("A::m()"); } }
impl<'a, 'b, 'c> M for &'a &'b &'c A { fn m(self) { println!("&&&A::m()"); } }
impl             RefM for             A { fn refm(&self) { println!("A::refm()"); } }
impl<'a, 'b, 'c> RefM for &'a &'b &'c A { fn refm(&self) { println!("&&&A::refm()"); } }

fn main() {
    // I'll use @ to denote left side of the dot operator
    (*X{val:42}).m();        // i32::refm() , self == @
    X{val:42}.m();           // X::m()      , self == @
    (&X{val:42}).m();        // &X::m()     , self == @
    (&&X{val:42}).m();       // &&X::m()    , self == @
    (&&&X{val:42}).m();      // &&&X:m()    , self == @
    (&&&&X{val:42}).m();     // &&&X::m()   , self == *@
    (&&&&&X{val:42}).m();    // &&&X::m()   , self == **@

    (*X{val:42}).refm();     // i32::refm() , self == @
    X{val:42}.refm();        // X::refm()   , self == @
    (&X{val:42}).refm();     // X::refm()   , self == *@
    (&&X{val:42}).refm();    // &X::refm()  , self == *@
    (&&&X{val:42}).refm();   // &&X::refm() , self == *@
    (&&&&X{val:42}).refm();  // &&&X::refm(), self == *@
    (&&&&&X{val:42}).refm(); // &&&X::refm(), self == **@

    Y{val:42}.refm();        // i32::refm() , self == *@
    Z{val:Y{val:42}}.refm(); // i32::refm() , self == **@

    A.m();                   // A::m()      , self == @
    // without the Copy trait, (&A).m() would be a compilation error:
    // cannot move out of borrowed content
    (&A).m();                // A::m()      , self == *@
    (&&A).m();               // &&&A::m()   , self == &@
    (&&&A).m();              // &&&A::m()   , self == @
    A.refm();                // A::refm()   , self == @
    (&A).refm();             // A::refm()   , self == *@
    (&&A).refm();            // A::refm()   , self == **@
    (&&&A).refm();           // &&&A::refm(), self == @
}
