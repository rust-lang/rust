//@aux-build:proc_macros.rs:proc-macro

#![deny(clippy::implicit_hasher)]
#![allow(unused)]

#[macro_use]
extern crate proc_macros;
use proc_macros::external;

use std::cmp::Eq;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash};

pub trait Foo<T>: Sized {
    fn make() -> (Self, Self);
}

impl<K: Hash + Eq, V> Foo<i8> for HashMap<K, V> {
    fn make() -> (Self, Self) {
        // OK, don't suggest to modify these
        let _: HashMap<i32, i32> = HashMap::new();
        let _: HashSet<i32> = HashSet::new();

        (HashMap::new(), HashMap::with_capacity(10))
    }
}
impl<K: Hash + Eq, V> Foo<i8> for (HashMap<K, V>,) {
    fn make() -> (Self, Self) {
        ((HashMap::new(),), (HashMap::with_capacity(10),))
    }
}
impl Foo<i16> for HashMap<String, String> {
    fn make() -> (Self, Self) {
        (HashMap::new(), HashMap::with_capacity(10))
    }
}

impl<K: Hash + Eq, V, S: BuildHasher + Default> Foo<i32> for HashMap<K, V, S> {
    fn make() -> (Self, Self) {
        (HashMap::default(), HashMap::with_capacity_and_hasher(10, S::default()))
    }
}
impl<S: BuildHasher + Default> Foo<i64> for HashMap<String, String, S> {
    fn make() -> (Self, Self) {
        (HashMap::default(), HashMap::with_capacity_and_hasher(10, S::default()))
    }
}

impl<T: Hash + Eq> Foo<i8> for HashSet<T> {
    fn make() -> (Self, Self) {
        (HashSet::new(), HashSet::with_capacity(10))
    }
}
impl Foo<i16> for HashSet<String> {
    fn make() -> (Self, Self) {
        (HashSet::new(), HashSet::with_capacity(10))
    }
}

impl<T: Hash + Eq, S: BuildHasher + Default> Foo<i32> for HashSet<T, S> {
    fn make() -> (Self, Self) {
        (HashSet::default(), HashSet::with_capacity_and_hasher(10, S::default()))
    }
}
impl<S: BuildHasher + Default> Foo<i64> for HashSet<String, S> {
    fn make() -> (Self, Self) {
        (HashSet::default(), HashSet::with_capacity_and_hasher(10, S::default()))
    }
}

pub fn foo(_map: &mut HashMap<i32, i32>, _set: &mut HashSet<i32>) {}

#[proc_macros::inline_macros]
pub mod gen {
    use super::*;
    inline! {
        impl<K: Hash + Eq, V> Foo<u8> for HashMap<K, V> {
            fn make() -> (Self, Self) {
                (HashMap::new(), HashMap::with_capacity(10))
            }
        }

        pub fn bar(_map: &mut HashMap<i32, i32>, _set: &mut HashSet<i32>) {}
    }
}

// When the macro is in a different file, the suggestion spans can't be combined properly
// and should not cause an ICE
// See #2707
#[macro_use]
#[path = "auxiliary/test_macro.rs"]
pub mod test_macro;
__implicit_hasher_test_macro!(impl<K, V> for HashMap<K, V> where V: test_macro::A);

// #4260
external! {
    pub fn f(input: &HashMap<u32, u32>) {}
}

// #7712
pub async fn election_vote(_data: HashMap<i32, i32>) {}

fn main() {}
