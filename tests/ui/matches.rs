#![feature(exclusive_range_pattern)]
#![warn(clippy::all)]
#![allow(unused, clippy::redundant_pattern_matching, clippy::too_many_lines)]
#![warn(clippy::match_same_arms)]

fn dummy() {}

fn ref_pats() {
    {
        let v = &Some(0);
        match v {
            &Some(v) => println!("{:?}", v),
            &None => println!("none"),
        }
        match v {
            // This doesn't trigger; we have a different pattern.
            &Some(v) => println!("some"),
            other => println!("other"),
        }
    }
    let tup = &(1, 2);
    match tup {
        &(v, 1) => println!("{}", v),
        _ => println!("none"),
    }
    // Special case: using `&` both in expr and pats.
    let w = Some(0);
    match &w {
        &Some(v) => println!("{:?}", v),
        &None => println!("none"),
    }
    // False positive: only wildcard pattern.
    let w = Some(0);
    match w {
        _ => println!("none"),
    }

    let a = &Some(0);
    if let &None = a {
        println!("none");
    }

    let b = Some(0);
    if let &None = &b {
        println!("none");
    }
}

fn match_wild_err_arm() {
    let x: Result<i32, &str> = Ok(3);

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => panic!("err"),
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => panic!(),
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {
            panic!();
        },
    }

    // Allowed when not with `panic!` block.
    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => println!("err"),
    }

    // Allowed when used with `unreachable!`.
    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => unreachable!(),
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => unreachable!(),
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {
            unreachable!();
        },
    }

    // No warning because of the guard.
    match x {
        Ok(x) if x * x == 64 => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => println!("err"),
    }

    // This used to be a false positive; see issue #1996.
    match x {
        Ok(3) => println!("ok"),
        Ok(x) if x * x == 64 => println!("ok 64"),
        Ok(_) => println!("ok"),
        Err(_) => println!("err"),
    }

    match (x, Some(1i32)) {
        (Ok(x), Some(_)) => println!("ok {}", x),
        (Ok(_), Some(x)) => println!("ok {}", x),
        _ => println!("err"),
    }

    // No warning; different types for `x`.
    match (x, Some(1.0f64)) {
        (Ok(x), Some(_)) => println!("ok {}", x),
        (Ok(_), Some(x)) => println!("ok {}", x),
        _ => println!("err"),
    }

    // Because of a bug, no warning was generated for this case before #2251.
    match x {
        Ok(_tmp) => println!("ok"),
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {
            unreachable!();
        },
    }
}

fn match_as_ref() {
    let owned: Option<()> = None;
    let borrowed: Option<&()> = match owned {
        None => None,
        Some(ref v) => Some(v),
    };

    let mut mut_owned: Option<()> = None;
    let borrow_mut: Option<&mut ()> = match mut_owned {
        None => None,
        Some(ref mut v) => Some(v),
    };
}

macro_rules! foo_variant(
    ($idx:expr) => (Foo::get($idx).unwrap())
);

enum Foo {
    A,
    B,
}

impl Foo {
    fn get(idx: u8) -> Option<&'static Self> {
        match idx {
            0 => Some(&Foo::A),
            1 => Some(&Foo::B),
            _ => None,
        }
    }
}

fn main() {
    // ICE #3719
    match foo_variant!(0) {
        &Foo::A => println!("A"),
        _ => println!("Wild"),
    }
}
