#![warn(clippy::if_same_then_else)]
#![allow(
    clippy::disallowed_names,
    clippy::eq_op,
    clippy::never_loop,
    clippy::no_effect,
    clippy::unused_unit,
    clippy::zero_divided_by_zero,
    clippy::branches_sharing_code,
    dead_code,
    unreachable_code
)]

struct Foo {
    bar: u8,
}

fn foo() -> bool {
    unimplemented!()
}

fn if_same_then_else() {
    if true {
        Foo { bar: 42 };
        0..10;
        ..;
        0..;
        ..10;
        0..=10;
        foo();
    } else {
        Foo { bar: 42 };
        0..10;
        ..;
        0..;
        ..10;
        0..=10;
        foo();
    }
    //~^^^^^^^^^^^^^^^^^ if_same_then_else

    if true {
        Foo { bar: 42 };
    } else {
        Foo { bar: 43 };
    }

    if true {
        ();
    } else {
        ()
    }

    if true {
        0..10;
    } else {
        0..=10;
    }

    if true {
        foo();
        foo();
    } else {
        foo();
    }

    let _ = if true { 0.0 } else { 0.0 };
    //~^ if_same_then_else

    let _ = if true { -0.0 } else { -0.0 };
    //~^ if_same_then_else

    let _ = if true { 0.0 } else { -0.0 };

    // Different NaNs
    let _ = if true { 0.0 / 0.0 } else { f32::NAN };

    if true {
        foo();
    }

    let _ = if true { 42 } else { 42 };
    //~^ if_same_then_else

    if true {
        let bar = if true { 42 } else { 43 };

        while foo() {
            break;
        }
        bar + 1;
    } else {
        let bar = if true { 42 } else { 43 };

        while foo() {
            break;
        }
        bar + 1;
    }
    //~^^^^^^^^^^^^^^^ if_same_then_else

    if true {
        let _ = match 42 {
            42 => 1,
            a if a > 0 => 2,
            10..=15 => 3,
            _ => 4,
        };
    } else if false {
        foo();
    } else if foo() {
        let _ = match 42 {
            42 => 1,
            a if a > 0 => 2,
            10..=15 => 3,
            _ => 4,
        };
    }
}

// Issue #2423. This was causing an ICE.
fn func() {
    if true {
        f(&[0; 62]);
        f(&[0; 4]);
        f(&[0; 3]);
    } else {
        f(&[0; 62]);
        f(&[0; 6]);
        f(&[0; 6]);
    }
}

fn f(val: &[u8]) {}

mod issue_5698 {
    fn mul_not_always_commutative(x: i32, y: i32) -> i32 {
        if x == 42 {
            x * y
        } else if x == 21 {
            y * x
        } else {
            0
        }
    }
}

mod issue_8836 {
    fn do_not_lint() {
        if true {
            todo!()
        } else {
            todo!()
        }
        if true {
            todo!();
        } else {
            todo!();
        }
        if true {
            unimplemented!()
        } else {
            unimplemented!()
        }
        if true {
            unimplemented!();
        } else {
            unimplemented!();
        }

        if true {
            println!("FOO");
            todo!();
        } else {
            println!("FOO");
            todo!();
        }

        if true {
            println!("FOO");
            unimplemented!();
        } else {
            println!("FOO");
            unimplemented!();
        }

        if true {
            println!("FOO");
            todo!()
        } else {
            println!("FOO");
            todo!()
        }

        if true {
            println!("FOO");
            unimplemented!()
        } else {
            println!("FOO");
            unimplemented!()
        }
    }
}

mod issue_11213 {
    fn reproducer(x: bool) -> bool {
        if x {
            0_u8.is_power_of_two()
        } else {
            0_u16.is_power_of_two()
        }
    }

    // a more obvious reproducer that shows
    // why the code above is problematic:
    fn v2(x: bool) -> bool {
        trait Helper {
            fn is_u8(&self) -> bool;
        }
        impl Helper for u8 {
            fn is_u8(&self) -> bool {
                true
            }
        }
        impl Helper for u16 {
            fn is_u8(&self) -> bool {
                false
            }
        }

        // this is certainly not the same code in both branches
        // it returns a different bool depending on the branch.
        if x { 0_u8.is_u8() } else { 0_u16.is_u8() }
    }

    fn do_lint(x: bool) -> bool {
        // but do lint if the type of the literal is the same
        if x {
            0_u8.is_power_of_two()
        } else {
            0_u8.is_power_of_two()
        }
        //~^^^^^ if_same_then_else
    }
}

fn main() {}
