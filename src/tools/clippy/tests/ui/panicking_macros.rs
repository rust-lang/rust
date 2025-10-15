#![allow(clippy::assertions_on_constants, clippy::eq_op, clippy::let_unit_value)]
#![warn(clippy::unimplemented, clippy::unreachable, clippy::todo, clippy::panic)]

extern crate core;

const _: () = {
    if 1 == 0 {
        panic!("A balanced diet means a cupcake in each hand");
    }
};

fn inline_const() {
    let _ = const {
        if 1 == 0 {
            panic!("When nothing goes right, go left")
        }
    };
}

fn panic() {
    let a = 2;
    panic!();
    //~^ panic

    panic!("message");
    //~^ panic

    panic!("{} {}", "panic with", "multiple arguments");
    //~^ panic

    let b = a + 2;
}

const fn panic_const() {
    let a = 2;
    panic!();
    //~^ panic

    panic!("message");
    //~^ panic

    panic!("{} {}", "panic with", "multiple arguments");
    //~^ panic

    let b = a + 2;
}

fn todo() {
    let a = 2;
    todo!();
    //~^ todo

    todo!("message");
    //~^ todo

    todo!("{} {}", "panic with", "multiple arguments");
    //~^ todo

    let b = a + 2;
}

fn unimplemented() {
    let a = 2;
    unimplemented!();
    //~^ unimplemented

    unimplemented!("message");
    //~^ unimplemented

    unimplemented!("{} {}", "panic with", "multiple arguments");
    //~^ unimplemented

    let b = a + 2;
}

fn unreachable() {
    let a = 2;
    unreachable!();
    //~^ unreachable

    unreachable!("message");
    //~^ unreachable

    unreachable!("{} {}", "panic with", "multiple arguments");
    //~^ unreachable

    let b = a + 2;
}

fn core_versions() {
    use core::{panic, todo, unimplemented, unreachable};
    panic!();
    //~^ panic

    todo!();
    //~^ todo

    unimplemented!();
    //~^ unimplemented

    unreachable!();
    //~^ unreachable
}

fn assert() {
    assert!(true);
    assert_eq!(true, true);
    assert_ne!(true, false);
}

fn assert_msg() {
    assert!(true, "this should not panic");
    assert_eq!(true, true, "this should not panic");
    assert_ne!(true, false, "this should not panic");
}

fn debug_assert() {
    debug_assert!(true);
    debug_assert_eq!(true, true);
    debug_assert_ne!(true, false);
}

fn debug_assert_msg() {
    debug_assert!(true, "test");
    debug_assert_eq!(true, true, "test");
    debug_assert_ne!(true, false, "test");
}

fn main() {
    panic();
    panic_const();
    todo();
    unimplemented();
    unreachable();
    core_versions();
    assert();
    assert_msg();
    debug_assert();
    debug_assert_msg();
}
