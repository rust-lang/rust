//@ revisions: nofallback fallback
//@ run-pass

#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unreachable_code)]
// Test various cases where we permit an unconstrained variable
// to fallback based on control-flow. In all of these cases,
// the type variable winds up being the target of both a `!` coercion
// and a coercion from a non-`!` variable, and hence falls back to `()`.
#![cfg_attr(fallback, feature(never_type, never_type_fallback))]

trait UnitDefault {
    fn default() -> Self;
}

impl UnitDefault for u32 {
    fn default() -> Self {
        0
    }
}

impl UnitDefault for () {
    fn default() -> () {
        panic!()
    }
}

fn assignment() {
    //[nofallback]~^ warn: this function depends on never type fallback being `()`
    //[nofallback]~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in Rust 2024 and in a future release in all editions!
    let x;

    if true {
        x = UnitDefault::default();
    } else {
        x = return;
    }
}

fn assignment_rev() {
    //[nofallback]~^ warn: this function depends on never type fallback being `()`
    //[nofallback]~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in Rust 2024 and in a future release in all editions!
    let x;

    if true {
        x = return;
    } else {
        x = UnitDefault::default();
    }
}

fn if_then_else() {
    let _x: () = if true {
        UnitDefault::default()
    } else {
        return;
    };
}

fn if_then_else_rev() {
    let _x: () = if true {
        return;
    } else {
        UnitDefault::default()
    };
}

fn match_arm() {
    let _x: () = match Ok(UnitDefault::default()) {
        Ok(v) => v,
        Err(()) => return,
    };
}

fn match_arm_rev() {
    let _x: () = match Ok(UnitDefault::default()) {
        Err(()) => return,
        Ok(v) => v,
    };
}

fn loop_break() {
    let _x: () = loop {
        if false {
            break return;
        } else {
            break UnitDefault::default();
        }
    };
}

fn loop_break_rev() {
    let _x: () = loop {
        if false {
            break return;
        } else {
            break UnitDefault::default();
        }
    };
}

fn main() {}
