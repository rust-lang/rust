// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

fn ident<T>(t: T) -> T {
    t
}

// EMIT_MIR terminators.direct_call.built.after.mir
#[custom_mir(dialect = "built")]
fn direct_call(x: i32) -> i32 {
    mir! {
        {
            Call(RET = ident(x), ReturnTo(retblock), UnwindContinue())
        }

        retblock = {
            Return()
        }
    }
}

// EMIT_MIR terminators.tail_call.built.after.mir
#[custom_mir(dialect = "built")]
fn tail_call(x: i32) -> i32 {
    mir! {
        let y;
        {
            y = x + 42;
            TailCall(ident(y))
        }
    }
}

// EMIT_MIR terminators.indirect_call.built.after.mir
#[custom_mir(dialect = "built")]
fn indirect_call(x: i32, f: fn(i32) -> i32) -> i32 {
    mir! {
        {
            Call(RET = f(x), ReturnTo(retblock), UnwindContinue())
        }
        retblock = {
            Return()
        }
    }
}

struct WriteOnDrop<'a>(&'a mut i32, i32);

impl<'a> Drop for WriteOnDrop<'a> {
    fn drop(&mut self) {
        *self.0 = self.1;
    }
}

// EMIT_MIR terminators.drop_first.built.after.mir
#[custom_mir(dialect = "built")]
fn drop_first<'a>(a: WriteOnDrop<'a>, b: WriteOnDrop<'a>) {
    mir! {
        {
            Drop(a, ReturnTo(retblock), UnwindContinue())
        }
        retblock = {
            a = Move(b);
            Return()
        }
    }
}

// EMIT_MIR terminators.drop_second.built.after.mir
#[custom_mir(dialect = "built")]
fn drop_second<'a>(a: WriteOnDrop<'a>, b: WriteOnDrop<'a>) {
    mir! {
        {
            Drop(b, ReturnTo(retblock), UnwindContinue())
        }
        retblock = {
            Return()
        }
    }
}

// EMIT_MIR terminators.assert_nonzero.built.after.mir
#[custom_mir(dialect = "built")]
fn assert_nonzero(a: i32) {
    mir! {
        {
            match a {
                0 => unreachable,
                _ => retblock
            }
        }
        unreachable = {
            Unreachable()
        }
        retblock = {
            Return()
        }
    }
}

fn main() {
    assert_eq!(direct_call(5), 5);
    assert_eq!(indirect_call(5, ident), 5);

    let mut a = 0;
    let mut b = 0;
    drop_first(WriteOnDrop(&mut a, 1), WriteOnDrop(&mut b, 1));
    assert_eq!((a, b), (1, 0));

    let mut a = 0;
    let mut b = 0;
    drop_second(WriteOnDrop(&mut a, 1), WriteOnDrop(&mut b, 1));
    assert_eq!((a, b), (0, 1));
}
