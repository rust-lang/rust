// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(dead_code)]

// use different types / traits to test all combinations

trait Const {
    const C: ();
}

trait StaticFn {
    fn sfn();
}

struct ConstStruct;
struct StaticFnStruct;

enum ConstEnum {}
enum StaticFnEnum {}

struct AliasedConstStruct;
struct AliasedStaticFnStruct;

enum AliasedConstEnum {}
enum AliasedStaticFnEnum {}

type AliasConstStruct    = AliasedConstStruct;
type AliasStaticFnStruct = AliasedStaticFnStruct;
type AliasConstEnum      = AliasedConstEnum;
type AliasStaticFnEnum   = AliasedStaticFnEnum;

macro_rules! impl_Const {($($T:ident),*) => {$(
    impl Const for $T {
        const C: () = ();
    }
)*}}

macro_rules! impl_StaticFn {($($T:ident),*) => {$(
    impl StaticFn for $T {
        fn sfn() {}
    }
)*}}

impl_Const!(ConstStruct, ConstEnum, AliasedConstStruct, AliasedConstEnum);
impl_StaticFn!(StaticFnStruct, StaticFnEnum, AliasedStaticFnStruct, AliasedStaticFnEnum);

fn main() {
    let _ = ConstStruct::C;
    let _ = ConstEnum::C;

    StaticFnStruct::sfn();
    StaticFnEnum::sfn();

    let _ = AliasConstStruct::C;
    let _ = AliasConstEnum::C;

    AliasStaticFnStruct::sfn();
    AliasStaticFnEnum::sfn();
}
