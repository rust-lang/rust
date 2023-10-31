//@run-rustfix
#![warn(clippy::match_single_binding)]
#![allow(
    unused,
    clippy::let_unit_value,
    clippy::no_effect,
    clippy::toplevel_ref_arg,
    clippy::uninlined_format_args,
    clippy::useless_vec
)]

struct Point {
    x: i32,
    y: i32,
}

fn coords() -> Point {
    Point { x: 1, y: 2 }
}

macro_rules! foo {
    ($param:expr) => {
        match $param {
            _ => println!("whatever"),
        }
    };
}

fn main() {
    let a = 1;
    let b = 2;
    let c = 3;
    // Lint
    match (a, b, c) {
        (x, y, z) => {
            println!("{} {} {}", x, y, z);
        },
    }
    // Lint
    match (a, b, c) {
        (x, y, z) => println!("{} {} {}", x, y, z),
    }
    // Ok
    foo!(a);
    // Ok
    match a {
        2 => println!("2"),
        _ => println!("Not 2"),
    }
    // Ok
    let d = Some(5);
    match d {
        Some(d) => println!("{}", d),
        _ => println!("None"),
    }
    // Lint
    match a {
        _ => println!("whatever"),
    }
    // Lint
    match a {
        _ => {
            let x = 29;
            println!("x has a value of {}", x);
        },
    }
    // Lint
    match a {
        _ => {
            let e = 5 * a;
            if e >= 5 {
                println!("e is superior to 5");
            }
        },
    }
    // Lint
    let p = Point { x: 0, y: 7 };
    match p {
        Point { x, y } => println!("Coords: ({}, {})", x, y),
    }
    // Lint
    match p {
        Point { x: x1, y: y1 } => println!("Coords: ({}, {})", x1, y1),
    }
    // Lint
    let x = 5;
    match x {
        ref r => println!("Got a reference to {}", r),
    }
    // Lint
    let mut x = 5;
    match x {
        ref mut mr => println!("Got a mutable reference to {}", mr),
    }
    // Lint
    let product = match coords() {
        Point { x, y } => x * y,
    };
    // Lint
    let v = vec![Some(1), Some(2), Some(3), Some(4)];
    #[allow(clippy::let_and_return)]
    let _ = v
        .iter()
        .map(|i| match i.unwrap() {
            unwrapped => unwrapped,
        })
        .collect::<Vec<u8>>();
    // Ok
    let x = 1;
    match x {
        #[cfg(disabled_feature)]
        0 => println!("Disabled branch"),
        _ => println!("Enabled branch"),
    }

    // Ok
    let x = 1;
    let y = 1;
    match match y {
        0 => 1,
        _ => 2,
    } {
        #[cfg(disabled_feature)]
        0 => println!("Array index start"),
        _ => println!("Not an array index start"),
    }

    // Lint
    let x = 1;
    match x {
        // =>
        _ => println!("Not an array index start"),
    }
}

fn issue_8723() {
    let (mut val, idx) = ("a b", 1);

    val = match val.split_at(idx) {
        (pre, suf) => {
            println!("{}", pre);
            suf
        },
    };

    let _ = val;
}

fn side_effects() {}

fn issue_9575() {
    let _ = || match side_effects() {
        _ => println!("Needs curlies"),
    };
}

fn issue_9725(r: Option<u32>) {
    match r {
        x => match x {
            Some(_) => {
                println!("Some");
            },
            None => {
                println!("None");
            },
        },
    };
}

fn issue_10447() -> usize {
    match 1 {
        _ => (),
    }

    let a = match 1 {
        _ => (),
    };

    match 1 {
        _ => side_effects(),
    }

    let b = match 1 {
        _ => side_effects(),
    };

    match 1 {
        _ => println!("1"),
    }

    let c = match 1 {
        _ => println!("1"),
    };

    let in_expr = [
        match 1 {
            _ => (),
        },
        match 1 {
            _ => side_effects(),
        },
        match 1 {
            _ => println!("1"),
        },
    ];

    2
}
