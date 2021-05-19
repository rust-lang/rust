#![allow(unreachable_code)]

pub trait MyHash {}

pub struct MyHashSet<T>(T);

impl<T> Eq for MyHashSet<T> where T: Eq + MyHash {}

impl<T> PartialEq for MyHashSet<T> where T: PartialEq + MyHash {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

pub struct CustomSet<T>(MyHashSet<T>);

impl<T> Eq for CustomSet<T> where T: Eq {}

impl<T> PartialEq for CustomSet<T> where T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 //~ERROR E0369
    }
}

fn main() {}
