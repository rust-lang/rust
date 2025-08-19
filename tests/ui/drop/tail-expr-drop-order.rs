//@ aux-build:edition-2021-macros.rs
//@ aux-build:edition-2024-macros.rs
//@ compile-flags: -Z validate-mir
//@ edition: 2024
//@ run-pass

#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]

#[macro_use]
extern crate edition_2021_macros;
#[macro_use]
extern crate edition_2024_macros;
use std::cell::RefCell;
use std::convert::TryInto;

#[derive(Default)]
struct DropOrderCollector(RefCell<Vec<u32>>);

struct LoudDrop<'a>(&'a DropOrderCollector, u32);

impl Drop for LoudDrop<'_> {
    fn drop(&mut self) {
        println!("{}", self.1);
        self.0.0.borrow_mut().push(self.1);
    }
}

impl DropOrderCollector {
    fn option_loud_drop(&self, n: u32) -> Option<LoudDrop<'_>> {
        Some(LoudDrop(self, n))
    }

    fn loud_drop(&self, n: u32) -> LoudDrop<'_> {
        LoudDrop(self, n)
    }

    fn assert_sorted(&self, expected: usize) {
        let result = self.0.borrow();
        assert_eq!(result.len(), expected);
        for i in 1..result.len() {
            assert!(
                result[i - 1] < result[i],
                "inversion at {} ({} followed by {})",
                i - 1,
                result[i - 1],
                result[i]
            );
        }
    }
}

fn edition_2021_around_2021() {
    let c = DropOrderCollector::default();
    let _ = edition_2021_block! {
        let a = c.loud_drop(1);
        edition_2021_block! {
            let b = c.loud_drop(0);
            c.loud_drop(2).1
        }
    };
    c.assert_sorted(3);
}

fn edition_2021_around_2024() {
    let c = DropOrderCollector::default();
    let _ = edition_2021_block! {
        let a = c.loud_drop(2);
        edition_2024_block! {
            let b = c.loud_drop(1);
            c.loud_drop(0).1
        }
    };
    c.assert_sorted(3);
}

fn edition_2024_around_2021() {
    let c = DropOrderCollector::default();
    let _ = edition_2024_block! {
        let a = c.loud_drop(2);
        edition_2021_block! {
            let b = c.loud_drop(0);
            c.loud_drop(1).1
        }
    };
    c.assert_sorted(3);
}

fn edition_2024_around_2024() {
    let c = DropOrderCollector::default();
    let _ = edition_2024_block! {
        let a = c.loud_drop(2);
        edition_2024_block! {
            let b = c.loud_drop(1);
            c.loud_drop(0).1
        }
    };
    c.assert_sorted(3);
}

fn main() {
    edition_2021_around_2021();
    edition_2021_around_2024();
    edition_2024_around_2021();
    edition_2024_around_2024();
}
