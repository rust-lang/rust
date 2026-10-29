//@ test-mir-pass: AssumeVariant

#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]

use std::cell::RefCell;
use std::intrinsics::mir::*;

pub enum Foo {
    A(u32),
    B(u32),
}

// EMIT_MIR downcast.basic.AssumeVariant.diff
pub fn basic(a: Foo) -> u32 {
    // CHECK-LABEL: fn basic
    // CHECK: = copy ((_1 is B).0: u32);
    // CHECK: = copy ((_1 is A).0: u32);
    match a {
        Foo::A(v) => v,
        Foo::B(v) => v,
    }
}

// EMIT_MIR downcast.unknown.AssumeVariant.diff
#[custom_mir(dialect = "runtime")]
pub fn unknown(x: Foo) -> u32 {
    // CHECK-LABEL: fn unknown
    // CHECK: = copy ((_1 as variant#0).0: u32);
    mir! {
        {
            RET = Field(Variant(x, 0, false), 0);
            Return()
        }
    }
}

// EMIT_MIR downcast.not_match.AssumeVariant.diff
#[custom_mir(dialect = "runtime")]
pub fn not_match(x: Foo) -> u32 {
    // CHECK-LABEL: fn not_match
    // CHECK: = copy ((_1 is variant#0).0: u32);
    // CHECK: = copy ((_1 as variant#0).0: u32);
    mir! {
        {
            let discr = Discriminant(x);
            match discr {
                0 => a,
                1 => b,
                _ => unreachable,
            }
        }
        a = {
            RET = Field(Variant(x, 0, false), 0);
            Return()
        }
        b = {
            RET = Field(Variant(x, 0, false), 0);
            Return()
        }
        unreachable = {
            Unreachable()
        }
    }
}

// EMIT_MIR downcast.nested.AssumeVariant.diff
pub fn nested(x: Option<Option<u8>>) -> u8 {
    // CHECK-LABEL: fn nested
    // CHECK: discriminant(((_1 is Some).0: std::option::Option<u8>));
    // CHECK: ((((_1 is Some).0: std::option::Option<u8>) is Some).0: u8);
    match x {
        Some(Some(v)) => v,
        _ => 0,
    }
}

enum Single {
    A(u32),
}

// EMIT_MIR downcast.single.AssumeVariant.diff
#[custom_mir(dialect = "runtime")]
pub fn single(v: Single) -> u32 {
    // CHECK-LABEL: fn single
    // CHECK: copy ((_1 is variant#0).0: u32);
    mir! {
        {
            RET = Field(Variant(v, 0, false), 0);
            Return()
        }
    }
}

enum Reference {
    A(RefCell<u32>),
    B(RefCell<u32>),
}

// EMIT_MIR downcast.reference.AssumeVariant.diff
pub fn reference(r: &Reference) -> &RefCell<u32> {
    // CHECK-LABEL: fn reference
    // CHECK: = &(((*_1) is B).0: std::cell::RefCell<u32>);
    // CHECK: = &(((*_1) is A).0: std::cell::RefCell<u32>);
    match r {
        Reference::A(c) => c,
        Reference::B(c) => c,
    }
}
