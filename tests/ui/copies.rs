#![feature(plugin, inclusive_range_syntax)]
#![plugin(clippy)]

#![allow(dead_code, no_effect, unnecessary_operation)]
#![allow(let_and_return)]
#![allow(needless_return)]
#![allow(unused_variables)]
#![allow(cyclomatic_complexity)]
#![allow(blacklisted_name)]
#![allow(collapsible_if)]
#![allow(zero_divided_by_zero, eq_op)]
#![allow(path_statements)]

fn bar<T>(_: T) {}
fn foo() -> bool { unimplemented!() }

struct Foo {
    bar: u8,
}

pub enum Abc {
    A,
    B,
    C,
}

#[deny(if_same_then_else)]
#[deny(match_same_arms)]
fn if_same_then_else() -> Result<&'static str, ()> {
    if true {
        Foo { bar: 42 };
        0..10;
        ..;
        0..;
        ..10;
        0...10;
        foo();
    }
    else {
        Foo { bar: 42 };
        0..10;
        ..;
        0..;
        ..10;
        0...10;
        foo();
    }

    if true {
        Foo { bar: 42 };
    }
    else {
        Foo { bar: 43 };
    }

    if true {
        ();
    }
    else {
        ()
    }

    if true {
        0..10;
    }
    else {
        0...10;
    }

    if true {
        foo();
        foo();
    }
    else {
        foo();
    }

    let _ = match 42 {
        42 => {
            foo();
            let mut a = 42 + [23].len() as i32;
            if true {
                a += 7;
            }
            a = -31-a;
            a
        }
        _ => {
            foo();
            let mut a = 42 + [23].len() as i32;
            if true {
                a += 7;
            }
            a = -31-a;
            a
        }
    };

    let _ = match Abc::A {
        Abc::A => 0,
        Abc::B => 1,
        _ => 0,
    };

    if true {
        foo();
    }

    let _ = if true {
        42
    }
    else {
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
    }
    else {
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
        let bar = if true {
            42
        }
        else {
            43
        };

        while foo() { break; }
        bar + 1;
    }
    else {
        let bar = if true {
            42
        }
        else {
            43
        };

        while foo() { break; }
        bar + 1;
    }

    if true {
        let _ = match 42 {
            42 => 1,
            a if a > 0 => 2,
            10...15 => 3,
            _ => 4,
        };
    }
    else if false {
        foo();
    }
    else if foo() {
        let _ = match 42 {
            42 => 1,
            a if a > 0 => 2,
            10...15 => 3,
            _ => 4,
        };
    }

    if true {
        if let Some(a) = Some(42) {}
    }
    else {
        if let Some(a) = Some(42) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    }
    else {
        if let (1, .., 3) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    }
    else {
        if let (.., 3) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    }
    else {
        if let (.., 4) = (1, 2, 3) {}
    }

    if true {
        if let (1, .., 3) = (1, 2, 3) {}
    }
    else {
        if let (.., 1, 3) = (1, 2, 3) {}
    }

    if true {
        if let Some(42) = None {}
    }
    else {
        if let Option::Some(42) = None {}
    }

    if true {
        if let Some(42) = None::<u8> {}
    }
    else {
        if let Some(42) = None {}
    }

    if true {
        if let Some(42) = None::<u8> {}
    }
    else {
        if let Some(42) = None::<u32> {}
    }

    if true {
        if let Some(a) = Some(42) {}
    }
    else {
        if let Some(a) = Some(43) {}
    }

    let _ = match 42 {
        42 => foo(),
        51 => foo(),
        _ => true,
    };

    let _ = match Some(42) {
        Some(_) => 24,
        None => 24,
    };

    let _ = match Some(42) {
        Some(foo) => 24,
        None => 24,
    };

    let _ = match Some(42) {
        Some(42) => 24,
        Some(a) => 24, // bindings are different
        None => 0,
    };

    let _ = match Some(42) {
        Some(a) if a > 0 => 24,
        Some(a) => 24, // one arm has a guard
        None => 0,
    };

    match (Some(42), Some(42)) {
        (Some(a), None) => bar(a),
        (None, Some(a)) => bar(a),
        _ => (),
    }

    match (Some(42), Some(42)) {
        (Some(a), ..) => bar(a),
        (.., Some(a)) => bar(a),
        _ => (),
    }

    match (1, 2, 3) {
        (1, .., 3) => 42,
        (.., 3) => 42,
        _ => 0,
    };

    let _ = if true {
        0.0
    } else {
        0.0
    };

    let _ = if true {
        -0.0
    } else {
        -0.0
    };

    let _ = if true {
        0.0
    } else {
        -0.0
    };

    // Different NaNs
    let _ = if true {
        0.0 / 0.0
    } else {
        std::f32::NAN
    };

    // Same NaNs
    let _ = if true {
        std::f32::NAN
    } else {
        std::f32::NAN
    };

    let _ = match Some(()) {
        Some(()) => 0.0,
        None => -0.0
    };

    match (Some(42), Some("")) {
        (Some(a), None) => bar(a),
        (None, Some(a)) => bar(a), // bindings have different types
        _ => (),
    }

    if true {
        try!(Ok("foo"));
    }
    else {
        try!(Ok("foo"));
    }

    if true {
        let foo = "";
        return Ok(&foo[0..]);
    }
    else if false {
        let foo = "bar";
        return Ok(&foo[0..]);
    }
    else {
        let foo = "";
        return Ok(&foo[0..]);
    }
}

#[deny(ifs_same_cond)]
#[allow(if_same_then_else)] // all empty blocks
fn ifs_same_cond() {
    let a = 0;
    let b = false;

    if b {
    }
    else if b {
    }

    if a == 1 {
    }
    else if a == 1 {
    }

    if 2*a == 1 {
    }
    else if 2*a == 2 {
    }
    else if 2*a == 1 {
    }
    else if a == 1 {
    }

    // See #659
    if cfg!(feature = "feature1-659") {
        1
    } else if cfg!(feature = "feature2-659") {
        2
    } else {
        3
    };

    let mut v = vec![1];
    if v.pop() == None { // ok, functions
    }
    else if v.pop() == None {
    }

    if v.len() == 42 { // ok, functions
    }
    else if v.len() == 42 {
    }
}

fn main() {}
