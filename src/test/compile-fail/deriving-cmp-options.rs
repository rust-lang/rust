// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Eq="foo")] //~ ERROR does not accept options with this syntax
struct S1 { a: int }

#[deriving(Eq(bad_option(x)))] //~ ERROR unrecognised option name
struct S2 { a: int }

#[deriving(Eq(bad_option))] //~ ERROR only accepts `test_order(...)` and `ignore(...)`
struct S3 { a: int }

#[deriving(Eq(reverse(a)), //~ ERROR does not allow `reverse`
           TotalEq(reverse(a)))] //~ ERROR does not allow `reverse`
struct S4 { a: int }

#[deriving(TotalEq(ignore(a)), //~ ERROR does not allow `ignore`
           TotalOrd(ignore(a)))] //~ ERROR does not allow `ignore`
struct S5 { a: int }

#[deriving(Eq(ignore(b)))] //~ ERROR field `b` does not exist
struct S6 { a: int }

#[deriving(Eq(ignore(a,a)))] //~ ERROR field `a` occurs more than once
struct S7 { a: int }

#[deriving(Eq(test_order(a), ignore(a)))] //~ ERROR in both `ignore` and `test_order`
struct S8 { a: int }

#[deriving(Ord(reverse(a), ignore(a)))] //~ ERROR in both `ignore` and `reverse`
struct S9 { a: int }

#[deriving(Eq(ignore()))] //~ ERROR cannot use options on a unit struct
struct Unit;

#[deriving(Eq(ignore()))] //~ ERROR cannot use options on a tuple struct
struct T(uint, uint);

#[deriving(Eq(ignore(A)))] //~ ERROR cannot use options on an enum
enum E { A, B }
