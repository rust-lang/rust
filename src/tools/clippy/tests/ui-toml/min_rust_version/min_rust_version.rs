#![allow(clippy::redundant_clone, clippy::unnecessary_operation)]
#![warn(clippy::manual_non_exhaustive, clippy::borrow_as_ptr, clippy::manual_bits)]

use std::mem::{size_of, size_of_val};
use std::ops::Deref;

mod enums {
    enum E {
        A,
        B,
        #[doc(hidden)]
        _C,
    }

    // user forgot to remove the marker
    #[non_exhaustive]
    enum Ep {
        A,
        B,
        #[doc(hidden)]
        _C,
    }
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

fn manual_strip_msrv() {
    let s = "hello, world!";
    if s.starts_with("hello, ") {
        assert_eq!(s["hello, ".len()..].to_uppercase(), "WORLD!");
    }
}

fn check_index_refutable_slice() {
    // This shouldn't trigger `clippy::index_refutable_slice` as the suggestion
    // would only be valid from 1.42.0 onward
    let slice: Option<&[u32]> = Some(&[1]);
    if let Some(slice) = slice {
        println!("{}", slice[0]);
    }
}

fn map_clone_suggest_copied() {
    // This should still trigger the lint but suggest `cloned()` instead of `copied()`
    let _: Option<u64> = Some(&16).map(|b| *b);
}

fn borrow_as_ptr() {
    let val = 1;
    let _p = &val as *const i32;

    let mut val_mut = 1;
    let _p_mut = &mut val_mut as *mut i32;
}

fn manual_bits() {
    size_of::<i8>() * 8;
    size_of_val(&0u32) * 8;
}

fn main() {
    option_as_ref_deref();
    match_like_matches();
    match_same_arms();
    match_same_arms2();
    manual_strip_msrv();
    check_index_refutable_slice();
    borrow_as_ptr();
}
