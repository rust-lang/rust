#![allow(clippy::redundant_clone)]
#![feature(custom_inner_attributes)]
#![clippy::msrv = "1.0.0"]

use std::ops::{Deref, RangeFrom};

fn approx_const() {
    let log2_10 = 3.321928094887362;
    let log10_2 = 0.301029995663981;
}

fn cloned_instead_of_copied() {
    let _ = [1].iter().cloned();
}

fn option_as_ref_deref() {
    let mut opt = Some(String::from("123"));

    let _ = opt.as_ref().map(String::as_str);
    let _ = opt.as_ref().map(|x| x.as_str());
    let _ = opt.as_mut().map(String::as_mut_str);
    let _ = opt.as_mut().map(|x| x.as_mut_str());
}

fn match_like_matches() {
    let _y = match Some(5) {
        Some(0) => true,
        _ => false,
    };
}

fn match_same_arms() {
    match (1, 2, 3) {
        (1, .., 3) => 42,
        (.., 3) => 42, //~ ERROR match arms have same body
        _ => 0,
    };
}

fn match_same_arms2() {
    let _ = match Some(42) {
        Some(_) => 24,
        None => 24, //~ ERROR match arms have same body
    };
}

pub fn manual_strip_msrv() {
    let s = "hello, world!";
    if s.starts_with("hello, ") {
        assert_eq!(s["hello, ".len()..].to_uppercase(), "WORLD!");
    }
}

pub fn redundant_fieldnames() {
    let start = 0;
    let _ = RangeFrom { start: start };
}

pub fn redundant_static_lifetime() {
    const VAR_ONE: &'static str = "Test constant #1";
}

pub fn checked_conversion() {
    let value: i64 = 42;
    let _ = value <= (u32::max_value() as i64) && value >= 0;
    let _ = value <= (u32::MAX as i64) && value >= 0;
}

pub struct FromOverInto(String);

impl Into<FromOverInto> for String {
    fn into(self) -> FromOverInto {
        FromOverInto(self)
    }
}

pub fn filter_map_next() {
    let a = ["1", "lol", "3", "NaN", "5"];

    #[rustfmt::skip]
    let _: Option<u32> = vec![1, 2, 3, 4, 5, 6]
        .into_iter()
        .filter_map(|x| {
            if x == 2 {
                Some(x * 2)
            } else {
                None
            }
        })
        .next();
}

#[allow(clippy::no_effect)]
#[allow(clippy::short_circuit_statement)]
#[allow(clippy::unnecessary_operation)]
pub fn manual_range_contains() {
    let x = 5;
    x >= 8 && x < 12;
}

pub fn use_self() {
    struct Foo {}

    impl Foo {
        fn new() -> Foo {
            Foo {}
        }
        fn test() -> Foo {
            Foo::new()
        }
    }
}

fn replace_with_default() {
    let mut s = String::from("foo");
    let _ = std::mem::replace(&mut s, String::default());
}

fn map_unwrap_or() {
    let opt = Some(1);

    // Check for `option.map(_).unwrap_or(_)` use.
    // Single line case.
    let _ = opt
        .map(|x| x + 1)
        // Should lint even though this call is on a separate line.
        .unwrap_or(0);
}

// Could be const
fn missing_const_for_fn() -> i32 {
    1
}

fn unnest_or_patterns() {
    struct TS(u8, u8);
    if let TS(0, x) | TS(1, x) = TS(0, 0) {}
}

fn main() {
    filter_map_next();
    checked_conversion();
    redundant_fieldnames();
    redundant_static_lifetime();
    option_as_ref_deref();
    match_like_matches();
    match_same_arms();
    match_same_arms2();
    manual_strip_msrv();
    manual_range_contains();
    use_self();
    replace_with_default();
    map_unwrap_or();
    missing_const_for_fn();
    unnest_or_patterns();
}

mod meets_msrv {
    #![feature(custom_inner_attributes)]
    #![clippy::msrv = "1.45.0"]

    fn main() {
        let s = "hello, world!";
        if s.starts_with("hello, ") {
            assert_eq!(s["hello, ".len()..].to_uppercase(), "WORLD!");
        }
    }
}

mod just_under_msrv {
    #![feature(custom_inner_attributes)]
    #![clippy::msrv = "1.46.0"]

    fn main() {
        let s = "hello, world!";
        if s.starts_with("hello, ") {
            assert_eq!(s["hello, ".len()..].to_uppercase(), "WORLD!");
        }
    }
}

mod just_above_msrv {
    #![feature(custom_inner_attributes)]
    #![clippy::msrv = "1.44.0"]

    fn main() {
        let s = "hello, world!";
        if s.starts_with("hello, ") {
            assert_eq!(s["hello, ".len()..].to_uppercase(), "WORLD!");
        }
    }
}
