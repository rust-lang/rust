// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type MyResultPriv<T> = Result<T, u16>;
pub type MyResultPub<T> = Result<T, u64>;

// @has private_type_alias/fn.get_result_priv.html '//pre' 'Result<u8, u16>'
pub fn get_result_priv() -> MyResultPriv<u8> {
    panic!();
}

// @has private_type_alias/fn.get_result_pub.html '//pre' 'MyResultPub<u32>'
pub fn get_result_pub() -> MyResultPub<u32> {
    panic!();
}

pub type PubRecursive = u16;
type PrivRecursive3 = u8;
type PrivRecursive2 = PubRecursive;
type PrivRecursive1 = PrivRecursive3;

// PrivRecursive1 is expanded twice and stops at u8
// PrivRecursive2 is expanded once and stops at public type alias PubRecursive
// @has private_type_alias/fn.get_result_recursive.html '//pre' '(u8, PubRecursive)'
pub fn get_result_recursive() -> (PrivRecursive1, PrivRecursive2) {
    panic!();
}

type MyLifetimePriv<'a> = &'a isize;

// @has private_type_alias/fn.get_lifetime_priv.html '//pre' "&'static isize"
pub fn get_lifetime_priv() -> MyLifetimePriv<'static> {
    panic!();
}
