// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:privacy-tuple-struct.rs
// ignore-fast

extern crate "privacy-tuple-struct" as other;

mod a {
    pub struct A(());
    pub struct B(int);
    pub struct C(pub int, int);
    pub struct D(pub int);

    fn test() {
        let a = A(());
        let b = B(2);
        let c = C(2, 3);
        let d = D(4);

        let A(()) = a;
        let A(_) = a;
        match a { A(()) => {} }
        match a { A(_) => {} }

        let B(_) = b;
        let B(_b) = b;
        match b { B(_) => {} }
        match b { B(_b) => {} }
        match b { B(1) => {} B(_) => {} }

        let C(_, _) = c;
        let C(_a, _) = c;
        let C(_, _b) = c;
        let C(_a, _b) = c;
        match c { C(_, _) => {} }
        match c { C(_a, _) => {} }
        match c { C(_, _b) => {} }
        match c { C(_a, _b) => {} }

        let D(_) = d;
        let D(_d) = d;
        match d { D(_) => {} }
        match d { D(_d) => {} }
        match d { D(1) => {} D(_) => {} }

        let a2 = A;
        let b2 = B;
        let c2 = C;
        let d2 = D;
    }
}

fn this_crate() {
    let a = a::A(()); //~ ERROR: cannot invoke tuple struct constructor
    let b = a::B(2); //~ ERROR: cannot invoke tuple struct constructor
    let c = a::C(2, 3); //~ ERROR: cannot invoke tuple struct constructor
    let d = a::D(4);

    let a::A(()) = a; //~ ERROR: field #1 of struct `a::A` is private
    let a::A(_) = a;
    match a { a::A(()) => {} } //~ ERROR: field #1 of struct `a::A` is private
    match a { a::A(_) => {} }

    let a::B(_) = b;
    let a::B(_b) = b; //~ ERROR: field #1 of struct `a::B` is private
    match b { a::B(_) => {} }
    match b { a::B(_b) => {} } //~ ERROR: field #1 of struct `a::B` is private
    match b { a::B(1) => {} a::B(_) => {} } //~ ERROR: field #1 of struct `a::B` is private

    let a::C(_, _) = c;
    let a::C(_a, _) = c;
    let a::C(_, _b) = c; //~ ERROR: field #2 of struct `a::C` is private
    let a::C(_a, _b) = c; //~ ERROR: field #2 of struct `a::C` is private
    match c { a::C(_, _) => {} }
    match c { a::C(_a, _) => {} }
    match c { a::C(_, _b) => {} } //~ ERROR: field #2 of struct `a::C` is private
    match c { a::C(_a, _b) => {} } //~ ERROR: field #2 of struct `a::C` is private

    let a::D(_) = d;
    let a::D(_d) = d;
    match d { a::D(_) => {} }
    match d { a::D(_d) => {} }
    match d { a::D(1) => {} a::D(_) => {} }

    let a2 = a::A; //~ ERROR: cannot invoke tuple struct constructor
    let b2 = a::B; //~ ERROR: cannot invoke tuple struct constructor
    let c2 = a::C; //~ ERROR: cannot invoke tuple struct constructor
    let d2 = a::D;
}

fn xcrate() {
    let a = other::A(()); //~ ERROR: cannot invoke tuple struct constructor
    let b = other::B(2); //~ ERROR: cannot invoke tuple struct constructor
    let c = other::C(2, 3); //~ ERROR: cannot invoke tuple struct constructor
    let d = other::D(4);

    let other::A(()) = a; //~ ERROR: field #1 of struct `privacy-tuple-struct::A` is private
    let other::A(_) = a;
    match a { other::A(()) => {} }
    //~^ ERROR: field #1 of struct `privacy-tuple-struct::A` is private
    match a { other::A(_) => {} }

    let other::B(_) = b;
    let other::B(_b) = b; //~ ERROR: field #1 of struct `privacy-tuple-struct::B` is private
    match b { other::B(_) => {} }
    match b { other::B(_b) => {} }
    //~^ ERROR: field #1 of struct `privacy-tuple-struct::B` is private
    match b { other::B(1) => {} other::B(_) => {} }
    //~^ ERROR: field #1 of struct `privacy-tuple-struct::B` is private

    let other::C(_, _) = c;
    let other::C(_a, _) = c;
    let other::C(_, _b) = c; //~ ERROR: field #2 of struct `privacy-tuple-struct::C` is private
    let other::C(_a, _b) = c; //~ ERROR: field #2 of struct `privacy-tuple-struct::C` is private
    match c { other::C(_, _) => {} }
    match c { other::C(_a, _) => {} }
    match c { other::C(_, _b) => {} }
    //~^ ERROR: field #2 of struct `privacy-tuple-struct::C` is private
    match c { other::C(_a, _b) => {} }
    //~^ ERROR: field #2 of struct `privacy-tuple-struct::C` is private

    let other::D(_) = d;
    let other::D(_d) = d;
    match d { other::D(_) => {} }
    match d { other::D(_d) => {} }
    match d { other::D(1) => {} other::D(_) => {} }

    let a2 = other::A; //~ ERROR: cannot invoke tuple struct constructor
    let b2 = other::B; //~ ERROR: cannot invoke tuple struct constructor
    let c2 = other::C; //~ ERROR: cannot invoke tuple struct constructor
    let d2 = other::D;
}

fn main() {}
