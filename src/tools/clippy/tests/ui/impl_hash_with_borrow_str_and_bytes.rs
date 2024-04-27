#![warn(clippy::impl_hash_borrow_with_str_and_bytes)]

use std::borrow::Borrow;
use std::hash::{Hash, Hasher};

struct ExampleType {
    data: String,
}

impl Hash for ExampleType {
    //~^ ERROR: can't
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Borrow<str> for ExampleType {
    fn borrow(&self) -> &str {
        &self.data
    }
}

impl Borrow<[u8]> for ExampleType {
    fn borrow(&self) -> &[u8] {
        self.data.as_bytes()
    }
}

struct ShouldNotRaiseForHash {}
impl Hash for ShouldNotRaiseForHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
    }
}

struct ShouldNotRaiseForBorrow {}
impl Borrow<str> for ShouldNotRaiseForBorrow {
    fn borrow(&self) -> &str {
        todo!();
    }
}
impl Borrow<[u8]> for ShouldNotRaiseForBorrow {
    fn borrow(&self) -> &[u8] {
        todo!();
    }
}

struct ShouldNotRaiseForHashBorrowStr {}
impl Hash for ShouldNotRaiseForHashBorrowStr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
    }
}
impl Borrow<str> for ShouldNotRaiseForHashBorrowStr {
    fn borrow(&self) -> &str {
        todo!();
    }
}

struct ShouldNotRaiseForHashBorrowSlice {}
impl Hash for ShouldNotRaiseForHashBorrowSlice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
    }
}

impl Borrow<[u8]> for ShouldNotRaiseForHashBorrowSlice {
    fn borrow(&self) -> &[u8] {
        todo!();
    }
}

#[derive(Hash)]
//~^ ERROR: can't
struct Derived {
    data: String,
}

impl Borrow<str> for Derived {
    fn borrow(&self) -> &str {
        self.data.as_str()
    }
}

impl Borrow<[u8]> for Derived {
    fn borrow(&self) -> &[u8] {
        self.data.as_bytes()
    }
}

struct GenericExampleType<T> {
    data: T,
}

impl<T: Hash> Hash for GenericExampleType<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Borrow<str> for GenericExampleType<String> {
    fn borrow(&self) -> &str {
        &self.data
    }
}

impl Borrow<[u8]> for GenericExampleType<&'static [u8]> {
    fn borrow(&self) -> &[u8] {
        self.data
    }
}

struct GenericExampleType2<T> {
    data: T,
}

impl Hash for GenericExampleType2<String> {
    //~^ ERROR: can't
    // this is correctly throwing an error for generic with concrete impl
    // for all 3 types
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Borrow<str> for GenericExampleType2<String> {
    fn borrow(&self) -> &str {
        &self.data
    }
}

impl Borrow<[u8]> for GenericExampleType2<String> {
    fn borrow(&self) -> &[u8] {
        self.data.as_bytes()
    }
}
