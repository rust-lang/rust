#![warn(clippy::if_same_then_else)]
#![allow(
    clippy::blacklisted_name,
    clippy::collapsible_if,
    clippy::cognitive_complexity,
    clippy::eq_op,
    clippy::needless_return,
    clippy::never_loop,
    clippy::no_effect,
    clippy::zero_divided_by_zero,
    clippy::unused_unit
)]

struct Foo {
    bar: u8,
}

fn foo() -> bool {
    unimplemented!()
}

fn if_same_then_else() -> Result<&'static str, ()> {
    if true {
        Foo { bar: 42 };
        0..10;
        ..;
        0..;
        ..10;
        0..=10;
        foo();
    } else {
        //~ ERROR same body as `if` block
        Foo { bar: 42 };
        0..10;
        ..;
        0..;
        ..10;
        0..=10;
        foo();
    }

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

    let _ = if true {
        0.0
    } else {
        //~ ERROR same body as `if` block
        0.0
    };

    let _ = if true {
        -0.0
    } else {
        //~ ERROR same body as `if` block
        -0.0
    };

    let _ = if true { 0.0 } else { -0.0 };

    // Different NaNs
    let _ = if true { 0.0 / 0.0 } else { std::f32::NAN };

    if true {
        foo();
    }

    let _ = if true {
        42
    } else {
        //~ ERROR same body as `if` block
        42
    };

    if true {
        for _ in &[42] {
            let foo: &Option<_> = &Some::<u8>(42);
            if true {
                break;
            } else {
                continue;
            }
        }
    } else {
        //~ ERROR same body as `if` block
        for _ in &[42] {
            let foo: &Option<_> = &Some::<u8>(42);
            if true {
                break;
            } else {
                continue;
            }
        }
    }

    if true {
        let bar = if true { 42 } else { 43 };

        while foo() {
            break;
        }
        bar + 1;
    } else {
        //~ ERROR same body as `if` block
        let bar = if true { 42 } else { 43 };

        while foo() {
            break;
        }
        bar + 1;
    }

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

    if true {
        if let Some(a) = Some(42) {}
    } else {
        //~ ERROR same body as `if` block
        if let Some(a) = Some(42) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    } else {
        //~ ERROR same body as `if` block
        if let (1, .., 3) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    } else {
        if let (.., 3) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    } else {
        if let (.., 4) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    } else {
        if let (.., 1, 3) = (1, 2, 3) {}
    }

    if true {
        if let Some(42) = None {}
    } else {
        if let Option::Some(42) = None {}
    }

    if true {
        if let Some(42) = None::<u8> {}
    } else {
        if let Some(42) = None {}
    }

    if true {
        if let Some(42) = None::<u8> {}
    } else {
        if let Some(42) = None::<u32> {}
    }

    if true {
        if let Some(a) = Some(42) {}
    } else {
        if let Some(a) = Some(43) {}
    }

    // Same NaNs
    let _ = if true {
        std::f32::NAN
    } else {
        //~ ERROR same body as `if` block
        std::f32::NAN
    };

    if true {
        try!(Ok("foo"));
    } else {
        //~ ERROR same body as `if` block
        try!(Ok("foo"));
    }

    if true {
        let foo = "";
        return Ok(&foo[0..]);
    } else if false {
        let foo = "bar";
        return Ok(&foo[0..]);
    } else {
        let foo = "";
        return Ok(&foo[0..]);
    }

    if true {
        let foo = "";
        return Ok(&foo[0..]);
    } else if false {
        let foo = "bar";
        return Ok(&foo[0..]);
    } else if true {
        let foo = "";
        return Ok(&foo[0..]);
    } else {
        let foo = "";
        return Ok(&foo[0..]);
    }

    // False positive `if_same_then_else`: `let (x, y)` vs. `let (y, x)`; see issue #3559.
    if true {
        let foo = "";
        let (x, y) = (1, 2);
        return Ok(&foo[x..y]);
    } else {
        let foo = "";
        let (y, x) = (1, 2);
        return Ok(&foo[x..y]);
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

fn main() {}
