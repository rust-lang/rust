// run-pass
#![feature(derive_skip)]
#![allow(unused)]

#[derive(PartialEq, Debug)]
struct SkipPartialEq {
    f1: usize,
    #[skip(PartialEq)]
    f2: usize,
}

#[derive(PartialEq, PartialOrd, Debug)]
struct SkipPartialOrd {
    #[skip(PartialOrd, PartialEq)]
    f1: usize,
    f2: usize,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
struct SkipOrd {
    #[skip(Ord, PartialOrd, PartialEq)]
    f1: usize,
    f2: usize,
}

#[derive(Hash, Debug)]
struct SkipHash {
    f1: usize,
    #[skip(Hash)]
    f2: usize,
}

#[derive(Debug)]
struct SkipDebug {
    f1: usize,
    #[skip(Debug)]
    f2: usize,
}

#[derive(Debug)]
struct SkipDebugTuple(#[skip(Debug)] usize);

fn main() {
    assert_eq!(SkipPartialEq { f1: 0, f2: 1 }, SkipPartialEq { f1: 0, f2: 5 });
    assert!(SkipPartialOrd { f1: 10, f2: 1 } <= SkipPartialOrd { f1: 0, f2: 5 });
    assert_eq!(SkipOrd { f1: 0, f2: 1 }.cmp(&SkipOrd { f1: 0, f2: 5 }), std::cmp::Ordering::Less);
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut h1 = DefaultHasher::new();
    SkipHash { f1: 0, f2: 1 }.hash(&mut h1);
    let h1 = h1.finish();
    let mut h2 = DefaultHasher::new();
    SkipHash { f1: 0, f2: 5 }.hash(&mut h2);
    let h2 = h2.finish();
    assert_eq!(h1, h2);
    assert_eq!(format!("{:?}", SkipDebug { f1: 0, f2: 5 }), "SkipDebug { f1: 0 }");
    assert_eq!(format!("{:?}", SkipDebugTuple(0)), "SkipDebugTuple");
}
