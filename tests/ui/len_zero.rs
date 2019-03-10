#![warn(clippy::len_without_is_empty, clippy::len_zero)]
#![allow(dead_code, unused)]

pub struct PubOne;

impl PubOne {
    pub fn len(self: &Self) -> isize {
        1
    }
}

impl PubOne {
    // A second impl for this struct -- the error span shouldn't mention this.
    pub fn irrelevant(self: &Self) -> bool {
        false
    }
}

// Identical to `PubOne`, but with an `allow` attribute on the impl complaining `len`.
pub struct PubAllowed;

#[allow(clippy::len_without_is_empty)]
impl PubAllowed {
    pub fn len(self: &Self) -> isize {
        1
    }
}

// No `allow` attribute on this impl block, but that doesn't matter -- we only require one on the
// impl containing `len`.
impl PubAllowed {
    pub fn irrelevant(self: &Self) -> bool {
        false
    }
}

struct NotPubOne;

impl NotPubOne {
    pub fn len(self: &Self) -> isize {
        // No error; `len` is pub but `NotPubOne` is not exported anyway.
        1
    }
}

struct One;

impl One {
    fn len(self: &Self) -> isize {
        // No error; `len` is private; see issue #1085.
        1
    }
}

pub trait PubTraitsToo {
    fn len(self: &Self) -> isize;
}

impl PubTraitsToo for One {
    fn len(self: &Self) -> isize {
        0
    }
}

trait TraitsToo {
    fn len(self: &Self) -> isize;
    // No error; `len` is private; see issue #1085.
}

impl TraitsToo for One {
    fn len(self: &Self) -> isize {
        0
    }
}

struct HasPrivateIsEmpty;

impl HasPrivateIsEmpty {
    pub fn len(self: &Self) -> isize {
        1
    }

    fn is_empty(self: &Self) -> bool {
        false
    }
}

pub struct HasIsEmpty;

impl HasIsEmpty {
    pub fn len(self: &Self) -> isize {
        1
    }

    fn is_empty(self: &Self) -> bool {
        false
    }
}

struct Wither;

pub trait WithIsEmpty {
    fn len(self: &Self) -> isize;
    fn is_empty(self: &Self) -> bool;
}

impl WithIsEmpty for Wither {
    fn len(self: &Self) -> isize {
        1
    }

    fn is_empty(self: &Self) -> bool {
        false
    }
}

pub struct HasWrongIsEmpty;

impl HasWrongIsEmpty {
    pub fn len(self: &Self) -> isize {
        1
    }

    pub fn is_empty(self: &Self, x: u32) -> bool {
        false
    }
}

pub trait Empty {
    fn is_empty(&self) -> bool;
}

pub trait InheritingEmpty: Empty {
    // Must not trigger `LEN_WITHOUT_IS_EMPTY`.
    fn len(&self) -> isize;
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

    let z: &TraitsToo = &y;
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

    let with_is_empty: &WithIsEmpty = &Wither;
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

// This used to ICE.
pub trait Foo: Sized {}

pub trait DependsOnFoo: Foo {
    fn len(&mut self) -> usize;
}
