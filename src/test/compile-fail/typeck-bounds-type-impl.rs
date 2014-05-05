// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Verify that built-in trait implementations are enforced recursively on
// a type.

#![feature(struct_inherit)]

trait DummyTrait {}

struct ShareStruct;

impl DummyTrait for ShareStruct {}
impl Share for ShareStruct {}

struct SendShareStruct {
    a: ShareStruct
}

impl DummyTrait for SendShareStruct {}
impl Share for SendShareStruct {}

impl Send for SendShareStruct {}
//~^ ERROR cannot implement a built-in trait on a type that doesn't fulfill such trait

enum ShareEnum {}
impl Share for ShareEnum {}

enum ShareSendEnum {}
impl Share for ShareSendEnum {}
impl Send for ShareSendEnum {}

enum SendShareEnum {
    MyShareEnumVariant(ShareEnum),
    MyShareStructVariant(ShareStruct),
    MyShareSendEnumVariant(ShareEnum, ShareSendEnum)
}

impl Share for SendShareEnum {}
impl Send for SendShareEnum {}
//~^ ERROR cannot implement a built-in trait on a type that doesn't fulfill such trait
//~^^ ERROR cannot implement a built-in trait on a type that doesn't fulfill such trait
//~^^^ ERROR cannot implement a built-in trait on a type that doesn't fulfill such trait

virtual struct Base {
    a: ShareStruct
}

struct InheritedShareStruct: Base;

impl Share for InheritedShareStruct {}
impl Send for InheritedShareStruct {}
//~^ Error cannot implement a built-in trait on a type that doesn't fulfill such trait

#[deriving(Send)]
//~^ Error cannot implement a built-in trait on a type that doesn't fulfill such trait
struct SendShareStruct2 {
    a: ShareStruct
}


type MyType = ||:'static -> ShareStruct;

impl Share for MyType {}
//~^ ERROR cannot implement built-in traits on types that are not struct / enum


struct ProcNoShareStruct {
    a: proc()
}

impl Share for ProcNoShareStruct {}
//~^ Error cannot implement a built-in trait on a type that doesn't fulfill such trait

struct ProcShareStruct {
    a: proc():Share
}

impl Share for ProcShareStruct {}

struct ClosureShareStruct {
    a: ||:'static+Share
}

impl Share for ClosureShareStruct {}

struct ClosureNotShareStruct {
    a: ||:'static
}

impl Share for ClosureNotShareStruct {}
//~^ Error cannot implement a built-in trait on a type that doesn't fulfill such trait

fn main() {}
