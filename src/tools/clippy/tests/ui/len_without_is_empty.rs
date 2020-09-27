#![warn(clippy::len_without_is_empty)]
#![allow(dead_code, unused)]

pub struct PubOne;

impl PubOne {
    pub fn len(&self) -> isize {
        1
    }
}

impl PubOne {
    // A second impl for this struct -- the error span shouldn't mention this.
    pub fn irrelevant(&self) -> bool {
        false
    }
}

// Identical to `PubOne`, but with an `allow` attribute on the impl complaining `len`.
pub struct PubAllowed;

#[allow(clippy::len_without_is_empty)]
impl PubAllowed {
    pub fn len(&self) -> isize {
        1
    }
}

// No `allow` attribute on this impl block, but that doesn't matter -- we only require one on the
// impl containing `len`.
impl PubAllowed {
    pub fn irrelevant(&self) -> bool {
        false
    }
}

pub trait PubTraitsToo {
    fn len(&self) -> isize;
}

impl PubTraitsToo for One {
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

struct NotPubOne;

impl NotPubOne {
    pub fn len(&self) -> isize {
        // No error; `len` is pub but `NotPubOne` is not exported anyway.
        1
    }
}

struct One;

impl One {
    fn len(&self) -> isize {
        // No error; `len` is private; see issue #1085.
        1
    }
}

trait TraitsToo {
    fn len(&self) -> isize;
    // No error; `len` is private; see issue #1085.
}

impl TraitsToo for One {
    fn len(&self) -> isize {
        0
    }
}

struct HasPrivateIsEmpty;

impl HasPrivateIsEmpty {
    pub fn len(&self) -> isize {
        1
    }

    fn is_empty(&self) -> bool {
        false
    }
}

struct Wither;

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

pub trait Empty {
    fn is_empty(&self) -> bool;
}

pub trait InheritingEmpty: Empty {
    // Must not trigger `LEN_WITHOUT_IS_EMPTY`.
    fn len(&self) -> isize;
}

// This used to ICE.
pub trait Foo: Sized {}

pub trait DependsOnFoo: Foo {
    fn len(&mut self) -> usize;
}

fn main() {}
