// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR aggregate_exprs.tuple.built.after.mir
#[custom_mir(dialect = "built")]
fn tuple() -> (i32, bool) {
    mir! {
        {
            RET = (1, true);
            Return()
        }
    }
}

// EMIT_MIR aggregate_exprs.array.built.after.mir
#[custom_mir(dialect = "built")]
fn array() -> [i32; 2] {
    mir! {
        let x: [i32; 2];
        let one: i32;
        {
            x = [42, 43];
            one = 1;
            x = [one, 2];
            RET = Move(x);
            Return()
        }
    }
}

struct Foo {
    a: i32,
    b: i32,
}

enum Bar {
    Foo(Foo, i32),
}

union Onion {
    neon: i32,
    noun: f32,
}

// EMIT_MIR aggregate_exprs.adt.built.after.mir
#[custom_mir(dialect = "built")]
fn adt() -> Onion {
    mir! {
        let one: i32;
        let x: Foo;
        let y: Bar;
        {
            one = 1;
            x = Foo {
                a: 1,
                b: 2,
            };
            y = Bar::Foo(Move(x), one);
            RET = Onion { neon: Field(Variant(y, 0), 1) };
            Return()
        }
    }
}

fn main() {
    assert_eq!(tuple(), (1, true));
    assert_eq!(array(), [1, 2]);
    assert_eq!(unsafe { adt().neon }, 1);
}
