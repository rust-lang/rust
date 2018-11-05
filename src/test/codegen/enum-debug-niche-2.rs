// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test depends on a patch that was committed to upstream LLVM
// before 7.0, then backported to the Rust LLVM fork.  It tests that
// optimized enum debug info accurately reflects the enum layout.

// ignore-tidy-linelength
// ignore-windows
// min-system-llvm-version 7.0

// compile-flags: -g -C no-prepopulate-passes

// CHECK: {{.*}}DICompositeType{{.*}}tag: DW_TAG_variant_part,{{.*}}size: 32,{{.*}}
// CHECK: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "Placeholder",{{.*}}extraData: i64 4294967295{{[,)].*}}
// CHECK: {{.*}}DIDerivedType{{.*}}tag: DW_TAG_member,{{.*}}name: "Error",{{.*}}extraData: i64 0{{[,)].*}}

#![feature(never_type)]
#![feature(nll)]

#[derive(Copy, Clone)]
pub struct Entity {
    private: std::num::NonZeroU32,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Declaration;

impl TypeFamily for Declaration {
    type Base = Base;
    type Placeholder = !;

    fn intern_base_data(_: BaseKind<Self>) {}
}

#[derive(Copy, Clone)]
pub struct Base;

pub trait TypeFamily: Copy + 'static {
    type Base: Copy;
    type Placeholder: Copy;

    fn intern_base_data(_: BaseKind<Self>);
}

#[derive(Copy, Clone)]
pub enum BaseKind<F: TypeFamily> {
    Named(Entity),
    Placeholder(F::Placeholder),
    Error,
}

pub fn main() {
    let x = BaseKind::Error::<Declaration>;
    let y = 7;
}
