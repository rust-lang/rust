// unit-test: SimplifyStaticSwitch

#![crate_type = "lib"]
#![feature(core_intrinsics, custom_mir)]

use std::hint;
use std::intrinsics::mir::*;
use std::ops::ControlFlow;

// EMIT_MIR simplify_static_switch.too_complex.SimplifyStaticSwitch.diff
pub fn too_complex(x: Result<i32, usize>) -> Option<i32> {
    match {
        match x {
            Ok(v) => ControlFlow::Continue(v),
            Err(r) => ControlFlow::Break(r),
        }
    } {
        ControlFlow::Continue(v) => Some(v),
        ControlFlow::Break(_) => None,
    }
}

// EMIT_MIR simplify_static_switch.custom_discr.SimplifyStaticSwitch.diff
pub fn custom_discr(x: bool) -> u8 {
    #[repr(u8)]
    enum CustomDiscr {
        A = 35,
        B = 73,
        C = 99,
    }

    match if x { CustomDiscr::A } else { CustomDiscr::B } {
        CustomDiscr::A => 5,
        _ => 13,
    }
}

pub enum Foo {
    A,
    B,
    C,
}

// Make sure we do not thread through loop headers, to avoid
// creating irreducible CFGs.
// EMIT_MIR simplify_static_switch.loop_header.SimplifyStaticSwitch.diff
pub fn loop_header() {
    let mut foo = Foo::A;
    loop {
        match foo {
            Foo::A => foo = Foo::B,
            Foo::B => foo = Foo::A,
            Foo::C => return,
        }
    }
}

// EMIT_MIR simplify_static_switch.opt_as_ref_unchecked.SimplifyStaticSwitch.diff
pub unsafe fn opt_as_ref_unchecked<T>(opt: &Option<T>) -> &T {
    let opt = match opt {
        Some(ref val) => Some(val),
        None => None,
    };
    match opt {
        Some(val) => val,
        None => unsafe { hint::unreachable_unchecked() },
    }
}

// EMIT_MIR simplify_static_switch.opt_as_mut_unchecked.SimplifyStaticSwitch.diff
pub unsafe fn opt_as_mut_unchecked<T>(opt: &mut Option<T>) -> &mut T {
    let opt = match opt {
        Some(ref mut val) => Some(val),
        None => None,
    };
    match opt {
        Some(val) => val,
        None => unsafe { hint::unreachable_unchecked() },
    }
}

// Make sure that we do not apply this opt if the aggregate is borrowed before
// being switched on.
// EMIT_MIR simplify_static_switch.borrowed_aggregate.SimplifyStaticSwitch.diff
pub fn borrowed_aggregate(cond: bool) -> bool {
    let mut foo = if cond {
        Foo::A
    } else {
        Foo::B
    };
    // `bar` could indirectly mutate `foo` so we cannot optimize.
    let bar = &mut foo;
    match foo {
        Foo::A => true,
        Foo::B | Foo::C => false,
    }
}

// Make sure that we do not apply this opt if the aggregate is mutated before
// being switched on.
// EMIT_MIR simplify_static_switch.mutated_aggregate.SimplifyStaticSwitch.diff
pub fn mutated_aggregate(cond: bool, bar: Foo) -> bool {
    let mut foo = if cond {
        Foo::A
    } else {
        Foo::B
    };
    // We no longer know what variant `foo` is.
    foo = bar;
    match foo {
        Foo::A => true,
        Foo::B | Foo::C => false,
    }
}

// Make sure that we do not apply this opt if the discriminant is borrowed before
// being switched on.
// EMIT_MIR simplify_static_switch.borrowed_discriminant.SimplifyStaticSwitch.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn borrowed_discriminant() -> bool {
    mir!(
        let x: Foo;
        {
            x = Foo::A;
            Goto(bb1)
        }
        bb1 = {
            let a = Discriminant(x);
            // `a` could be indirectly mutated through `b`.
            let b = &mut a;
            match a {
                0 => bb2,
                _ => bb3,
            }
        }
        bb2 = {
            RET = true;
            Return()
        }
        bb3 = {
            RET = false;
            Return()
        }
    )
}

// Make sure that we do not apply this opt if the discriminant is mutated
// before we switch on it.
// EMIT_MIR simplify_static_switch.mutated_discriminant.SimplifyStaticSwitch.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn mutated_discriminant(b: isize) -> bool {
    mir!(
        let x: Foo;
        {
            x = Foo::A;
            Goto(bb1)
        }
        bb1 = {
            let a = Discriminant(x);
            // We no longer know what discriminant `a` is.
            a = b;
            match a {
                0 => bb2,
                _ => bb3,
            }
        }
        bb2 = {
            RET = true;
            Return()
        }
        bb3 = {
            RET = false;
            Return()
        }
    )
}
