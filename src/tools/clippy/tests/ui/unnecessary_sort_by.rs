// run-rustfix

#![allow(clippy::stable_sort_primitive)]

use std::cmp::Reverse;

fn unnecessary_sort_by() {
    fn id(x: isize) -> isize {
        x
    }

    let mut vec: Vec<isize> = vec![3, 6, 1, 2, 5];
    // Forward examples
    vec.sort_by(|a, b| a.cmp(b));
    vec.sort_unstable_by(|a, b| a.cmp(b));
    vec.sort_by(|a, b| (a + 5).abs().cmp(&(b + 5).abs()));
    vec.sort_unstable_by(|a, b| id(-a).cmp(&id(-b)));
    // Reverse examples
    vec.sort_by(|a, b| b.cmp(a));
    vec.sort_by(|a, b| (b + 5).abs().cmp(&(a + 5).abs()));
    vec.sort_unstable_by(|a, b| id(-b).cmp(&id(-a)));
    // Negative examples (shouldn't be changed)
    let c = &7;
    vec.sort_by(|a, b| (b - a).cmp(&(a - b)));
    vec.sort_by(|_, b| b.cmp(&5));
    vec.sort_by(|_, b| b.cmp(c));
    vec.sort_unstable_by(|a, _| a.cmp(c));

    // Ignore vectors of references
    let mut vec: Vec<&&&isize> = vec![&&&3, &&&6, &&&1, &&&2, &&&5];
    vec.sort_by(|a, b| (***a).abs().cmp(&(***b).abs()));
    vec.sort_unstable_by(|a, b| (***a).abs().cmp(&(***b).abs()));
    vec.sort_by(|a, b| b.cmp(a));
    vec.sort_unstable_by(|a, b| b.cmp(a));
}

// Do not suggest returning a reference to the closure parameter of `Vec::sort_by_key`
mod issue_5754 {
    #[derive(Clone, Copy)]
    struct Test(usize);

    #[derive(PartialOrd, Ord, PartialEq, Eq)]
    struct Wrapper<'a>(&'a usize);

    impl Test {
        fn name(&self) -> &usize {
            &self.0
        }

        fn wrapped(&self) -> Wrapper<'_> {
            Wrapper(&self.0)
        }
    }

    pub fn test() {
        let mut args: Vec<Test> = vec![];

        // Forward
        args.sort_by(|a, b| a.name().cmp(b.name()));
        args.sort_by(|a, b| a.wrapped().cmp(&b.wrapped()));
        args.sort_unstable_by(|a, b| a.name().cmp(b.name()));
        args.sort_unstable_by(|a, b| a.wrapped().cmp(&b.wrapped()));
        // Reverse
        args.sort_by(|a, b| b.name().cmp(a.name()));
        args.sort_by(|a, b| b.wrapped().cmp(&a.wrapped()));
        args.sort_unstable_by(|a, b| b.name().cmp(a.name()));
        args.sort_unstable_by(|a, b| b.wrapped().cmp(&a.wrapped()));
    }
}

// `Vec::sort_by_key` closure parameter is `F: FnMut(&T) -> K`
// The suggestion is destructuring T and we know T is not a reference, so test that non-Copy T are
// not linted.
mod issue_6001 {
    struct Test(String);

    impl Test {
        // Return an owned type so that we don't hit the fix for 5754
        fn name(&self) -> String {
            self.0.clone()
        }
    }

    pub fn test() {
        let mut args: Vec<Test> = vec![];

        // Forward
        args.sort_by(|a, b| a.name().cmp(&b.name()));
        args.sort_unstable_by(|a, b| a.name().cmp(&b.name()));
        // Reverse
        args.sort_by(|a, b| b.name().cmp(&a.name()));
        args.sort_unstable_by(|a, b| b.name().cmp(&a.name()));
    }
}

fn main() {
    unnecessary_sort_by();
    issue_5754::test();
    issue_6001::test();
}
