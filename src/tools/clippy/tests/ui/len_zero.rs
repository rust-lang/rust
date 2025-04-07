#![warn(clippy::len_zero)]
#![allow(
    dead_code,
    unused,
    clippy::needless_if,
    clippy::len_without_is_empty,
    clippy::const_is_empty
)]

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
        //~^ len_zero
        println!("This should not happen!");
    }

    if "".len() == 0 {}
    //~^ len_zero

    let s = "Hello, world!";
    let s1 = &s;
    let s2 = &s1;
    let s3 = &s2;
    let s4 = &s3;
    let s5 = &s4;
    let s6 = &s5;
    println!("{}", *s1 == "");
    //~^ comparison_to_empty
    println!("{}", **s2 == "");
    //~^ comparison_to_empty
    println!("{}", ***s3 == "");
    //~^ comparison_to_empty
    println!("{}", ****s4 == "");
    //~^ comparison_to_empty
    println!("{}", *****s5 == "");
    //~^ comparison_to_empty
    println!("{}", ******(s6) == "");
    //~^ comparison_to_empty

    let d2s = DerefToDerefToString {};
    println!("{}", &**d2s == "");
    //~^ comparison_to_empty

    println!("{}", std::borrow::Cow::Borrowed("") == "");
    //~^ comparison_to_empty

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
        //~^ len_zero
        println!("Or this!");
    }
    if has_is_empty.len() != 0 {
        //~^ len_zero
        println!("Or this!");
    }
    if has_is_empty.len() > 0 {
        //~^ len_zero
        println!("Or this!");
    }
    if has_is_empty.len() < 1 {
        //~^ len_zero
        println!("Or this!");
    }
    if has_is_empty.len() >= 1 {
        //~^ len_zero
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
        //~^ len_zero
        println!("Or this!");
    }
    if 0 != has_is_empty.len() {
        //~^ len_zero
        println!("Or this!");
    }
    if 0 < has_is_empty.len() {
        //~^ len_zero
        println!("Or this!");
    }
    if 1 <= has_is_empty.len() {
        //~^ len_zero
        println!("Or this!");
    }
    if 1 > has_is_empty.len() {
        //~^ len_zero
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
        //~^ len_zero
        println!("Or this!");
    }
    assert!(!with_is_empty.is_empty());

    let has_wrong_is_empty = HasWrongIsEmpty;
    if has_wrong_is_empty.len() == 0 {
        // No error; `HasWrongIsEmpty` does not have `.is_empty()`.
        println!("Or this!");
    }

    // issue #10529
    (has_is_empty.len() > 0).then(|| println!("This can happen."));
    //~^ len_zero
    (has_is_empty.len() == 0).then(|| println!("Or this!"));
    //~^ len_zero
}

fn test_slice(b: &[u8]) {
    if b.len() != 0 {}
    //~^ len_zero
}

// issue #11992
fn binop_with_macros() {
    macro_rules! len {
        ($seq:ident) => {
            $seq.len()
        };
    }

    macro_rules! compare_to {
        ($val:literal) => {
            $val
        };
        ($val:expr) => {{ $val }};
    }

    macro_rules! zero {
        () => {
            0
        };
    }

    let has_is_empty = HasIsEmpty;
    // Don't lint, suggesting changes might break macro compatibility.
    (len!(has_is_empty) > 0).then(|| println!("This can happen."));
    // Don't lint, suggesting changes might break macro compatibility.
    if len!(has_is_empty) == 0 {}
    // Don't lint
    if has_is_empty.len() == compare_to!(if true { 0 } else { 1 }) {}
    // This is fine
    if has_is_empty.len() == compare_to!(1) {}

    if has_is_empty.len() == compare_to!(0) {}
    //~^ len_zero
    if has_is_empty.len() == zero!() {}
    //~^ len_zero

    (compare_to!(0) < has_is_empty.len()).then(|| println!("This can happen."));
    //~^ len_zero
}

fn no_infinite_recursion() -> bool {
    struct S;

    impl Deref for S {
        type Target = Self;
        fn deref(&self) -> &Self::Target {
            self
        }
    }

    impl PartialEq<&'static str> for S {
        fn eq(&self, _other: &&'static str) -> bool {
            false
        }
    }

    // Do not crash while checking if S implements `.is_empty()`
    S == ""
}
