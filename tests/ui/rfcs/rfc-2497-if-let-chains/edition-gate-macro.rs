//@ revisions: edition2021 edition2024
//@ compile-flags: -Z lint-mir -Z validate-mir
//@ [edition2021] edition: 2021
//@ [edition2024] edition: 2024
//@ aux-build:macro-in-2021.rs
//@ aux-build:macro-in-2024.rs
//@ run-pass

#![allow(dead_code)]

use std::unreachable as never;
use std::cell::RefCell;
use std::convert::TryInto;

// Compiletest doesn't specify the needed --extern flags to make `extern crate` unneccessary
extern crate macro_in_2021;
extern crate macro_in_2024;

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
    fn print(&self, n: u32) {
        println!("{n}");
        self.0.borrow_mut().push(n)
    }
    fn some_loud(&self, n: u32) -> Option<LoudDrop> {
        Some(LoudDrop(self, n))
    }

    #[track_caller]
    fn validate(self) {
        assert!(
            self.0
                .into_inner()
                .into_iter()
                .enumerate()
                .all(|(idx, item)| idx + 1 == item.try_into().unwrap())
        );
    }
    fn with_macro_2021(self) {
        // Edition 2021 drop behaviour
        macro_in_2021::make_if!((let None = self.some_loud(2)) { never!() } {self.print(1) });
        macro_in_2021::make_if!(let (self.some_loud(4)) { never!() } { self.print(3) });
        self.validate();
    }
    fn with_macro_2024(self) {
        // Edition 2024 drop behaviour
        macro_in_2024::make_if!((let None = self.some_loud(1)) { never!() } { self.print(2) });
        macro_in_2024::make_if!(let (self.some_loud(3)) { never!() } { self.print(4) });
        self.validate();
    }
}

fn main() {
    // 2021 drop order if it's a 2021 macro creating the `if`
    // 2024 drop order if it's a 2024 macro creating the `if`

    // Compare this with edition-gate-macro-error.rs: We want to avoid exposing 2021 drop order,
    // because it can create bad MIR (issue #104843)
    // This test doesn't contain any let chains at all: it should be understood
    // in combination with `edition-gate-macro-error.rs`

    DropOrderCollector::default().with_macro_2021();
    DropOrderCollector::default().with_macro_2024();

}
