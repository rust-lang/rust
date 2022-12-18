// run-rustfix

#![warn(clippy::len_zero)]
#![allow(dead_code, unused, clippy::len_without_is_empty)]

extern crate core;
use core::ops::Deref;

pub struct One;
struct Wither;

trait TraitsToo {
    fn len(&self) -> isize;
    // No error; `len` is private; see issue #1085.
}

impl TraitsToo for One {
    fn len(&self) -> isize {
        0
    }
}

pub struct HasIsEmpty;

impl HasIsEmpty {
    pub fn len(&self) -> isize {
        1
    }

    fn is_empty(&self) -> bool {
        false
    }
}

pub struct HasWrongIsEmpty;

impl HasWrongIsEmpty {
    pub fn len(&self) -> isize {
        1
    }

    pub fn is_empty(&self, x: u32) -> bool {
        false
    }
}

pub trait WithIsEmpty {
    fn len(&self) -> isize;
    fn is_empty(&self) -> bool;
}

impl WithIsEmpty for Wither {
    fn len(&self) -> isize {
        1
    }

    fn is_empty(&self) -> bool {
        false
    }
}

struct DerefToDerefToString;

impl Deref for DerefToDerefToString {
    type Target = DerefToString;

    fn deref(&self) -> &Self::Target {
        &DerefToString {}
    }
}

struct DerefToString;

impl Deref for DerefToString {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        "Hello, world!"
    }
}

fn main() {
    let x = [1, 2];
    if x.len() == 0 {
        println!("This should not happen!");
    }

    if "".len() == 0 {}

    let s = "Hello, world!";
    let s1 = &s;
    let s2 = &s1;
    let s3 = &s2;
    let s4 = &s3;
    let s5 = &s4;
    let s6 = &s5;
    println!("{}", *s1 == "");
    println!("{}", **s2 == "");
    println!("{}", ***s3 == "");
    println!("{}", ****s4 == "");
    println!("{}", *****s5 == "");
    println!("{}", ******(s6) == "");

    let d2s = DerefToDerefToString {};
    println!("{}", &**d2s == "");

    let y = One;
    if y.len() == 0 {
        // No error; `One` does not have `.is_empty()`.
        println!("This should not happen either!");
    }

    let z: &dyn TraitsToo = &y;
    if z.len() > 0 {
        // No error; `TraitsToo` has no `.is_empty()` method.
        println!("Nor should this!");
    }

    let has_is_empty = HasIsEmpty;
    if has_is_empty.len() == 0 {
        println!("Or this!");
    }
    if has_is_empty.len() != 0 {
        println!("Or this!");
    }
    if has_is_empty.len() > 0 {
        println!("Or this!");
    }
    if has_is_empty.len() < 1 {
        println!("Or this!");
    }
    if has_is_empty.len() >= 1 {
        println!("Or this!");
    }
    if has_is_empty.len() > 1 {
        // No error.
        println!("This can happen.");
    }
    if has_is_empty.len() <= 1 {
        // No error.
        println!("This can happen.");
    }
    if 0 == has_is_empty.len() {
        println!("Or this!");
    }
    if 0 != has_is_empty.len() {
        println!("Or this!");
    }
    if 0 < has_is_empty.len() {
        println!("Or this!");
    }
    if 1 <= has_is_empty.len() {
        println!("Or this!");
    }
    if 1 > has_is_empty.len() {
        println!("Or this!");
    }
    if 1 < has_is_empty.len() {
        // No error.
        println!("This can happen.");
    }
    if 1 >= has_is_empty.len() {
        // No error.
        println!("This can happen.");
    }
    assert!(!has_is_empty.is_empty());

    let with_is_empty: &dyn WithIsEmpty = &Wither;
    if with_is_empty.len() == 0 {
        println!("Or this!");
    }
    assert!(!with_is_empty.is_empty());

    let has_wrong_is_empty = HasWrongIsEmpty;
    if has_wrong_is_empty.len() == 0 {
        // No error; `HasWrongIsEmpty` does not have `.is_empty()`.
        println!("Or this!");
    }
}

fn test_slice(b: &[u8]) {
    if b.len() != 0 {}
}
