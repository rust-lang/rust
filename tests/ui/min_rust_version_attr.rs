#![allow(clippy::redundant_clone)]
#![feature(custom_inner_attributes)]
#![clippy::msrv = "1.0.0"]

use std::ops::Deref;

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

fn main() {
    option_as_ref_deref();
    match_like_matches();
    match_same_arms();
    match_same_arms2();
    manual_strip_msrv();
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
