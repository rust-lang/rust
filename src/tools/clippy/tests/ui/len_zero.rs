// run-rustfix

#![warn(clippy::len_zero)]
#![allow(dead_code, unused, clippy::len_without_is_empty)]

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

fn main() {
    let x = [1, 2];
    if x.len() == 0 {
        println!("This should not happen!");
    }

    if "".len() == 0 {}

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

mod issue_3807 {
    // Avoid suggesting changes to ranges if the user did not enable `range_is_empty`.
    // See https://github.com/rust-lang/rust/issues/48111#issuecomment-445132965
    fn no_suggestion() {
        let _ = (0..42).len() == 0;
    }
}
