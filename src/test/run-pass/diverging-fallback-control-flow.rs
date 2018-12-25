#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unreachable_code)]
// Test various cases where we permit an unconstrained variable
// to fallback based on control-flow.
//
// These represent current behavior, but are pretty dubious.  I would
// like to revisit these and potentially change them. --nmatsakis

#![feature(never_type)]

trait BadDefault {
    fn default() -> Self;
}

impl BadDefault for u32 {
    fn default() -> Self {
        0
    }
}

impl BadDefault for ! {
    fn default() -> ! {
        panic!()
    }
}

fn assignment() {
    let x;

    if true {
        x = BadDefault::default();
    } else {
        x = return;
    }
}

fn assignment_rev() {
    let x;

    if true {
        x = return;
    } else {
        x = BadDefault::default();
    }
}

fn if_then_else() {
    let _x = if true {
        BadDefault::default()
    } else {
        return;
    };
}

fn if_then_else_rev() {
    let _x = if true {
        return;
    } else {
        BadDefault::default()
    };
}

fn match_arm() {
    let _x = match Ok(BadDefault::default()) {
        Ok(v) => v,
        Err(()) => return,
    };
}

fn match_arm_rev() {
    let _x = match Ok(BadDefault::default()) {
        Err(()) => return,
        Ok(v) => v,
    };
}

fn loop_break() {
    let _x = loop {
        if false {
            break return;
        } else {
            break BadDefault::default();
        }
    };
}

fn loop_break_rev() {
    let _x = loop {
        if false {
            break return;
        } else {
            break BadDefault::default();
        }
    };
}

fn main() { }
