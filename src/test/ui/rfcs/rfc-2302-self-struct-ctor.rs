// run-pass

#![allow(dead_code)]

use std::fmt::Display;

struct ST1(i32, i32);

impl ST1 {
    fn new() -> Self {
        ST1(0, 1)
    }

    fn ctor() -> Self {
        Self(1,2)         // Self as a constructor
    }

    fn pattern(self) {
        match self {
            Self(x, y) => println!("{} {}", x, y), // Self as a pattern
        }
    }
}

struct ST2<T>(T); // With type parameter

impl<T> ST2<T> where T: Display {

    fn ctor(v: T) -> Self {
        Self(v)
    }

    fn pattern(&self) {
        match self {
            Self(ref v) => println!("{}", v),
        }
    }
}

struct ST3<'a>(&'a i32); // With lifetime parameter

impl<'a> ST3<'a> {

    fn ctor(v: &'a i32) -> Self {
        Self(v)
    }

    fn pattern(self) {
        let Self(ref v) = self;
        println!("{}", v);
    }
}

struct ST4(usize);

impl ST4 {
    fn map(opt: Option<usize>) -> Option<Self> {
        opt.map(Self)     // use `Self` as a function passed somewhere
    }
}

struct ST5;               // unit struct

impl ST5 {
    fn ctor() -> Self {
        Self               // `Self` as a unit struct value
    }

    fn pattern(self) -> Self {
        match self {
            Self => Self,   // `Self` as a unit struct value for matching
        }
    }
}

struct ST6(i32);
type T = ST6;
impl T {
    fn ctor() -> Self {
        ST6(1)
    }

    fn type_alias(self) {
        let Self(_x) = match self { Self(x) => Self(x) };
        let _opt: Option<Self> = Some(0).map(Self);
    }
}

struct ST7<T1, T2>(T1, T2);

impl ST7<i32, usize> {

    fn ctor() -> Self {
        Self(1, 2)
    }

    fn pattern(self) -> Self {
        match self {
            Self(x, y) => Self(x, y),
        }
    }
}

fn main() {
    let v1 = ST1::ctor();
    v1.pattern();

    let v2 = ST2::ctor(10);
    v2.pattern();

    let local = 42;
    let v3 = ST3::ctor(&local);
    v3.pattern();

    let v4 = Some(1usize);
    let _ = ST4::map(v4);

    let v5 = ST5::ctor();
    v5.pattern();

    let v6 = ST6::ctor();
    v6.type_alias();

    let v7 = ST7::<i32, usize>::ctor();
    let r = v7.pattern();
    println!("{} {}", r.0, r.1)
}
