#![allow(clippy::stable_sort_primitive, clippy::useless_vec)]

use std::cell::Ref;

fn unnecessary_sort_by() {
    fn id(x: isize) -> isize {
        x
    }

    let mut vec: Vec<isize> = vec![3, 6, 1, 2, 5];
    // Forward examples
    vec.sort_by(|a, b| a.cmp(b));
    //~^ unnecessary_sort_by
    vec.sort_unstable_by(|a, b| a.cmp(b));
    //~^ unnecessary_sort_by
    vec.sort_by(|a, b| (a + 5).abs().cmp(&(b + 5).abs()));
    //~^ unnecessary_sort_by
    vec.sort_unstable_by(|a, b| id(-a).cmp(&id(-b)));
    //~^ unnecessary_sort_by
    // Reverse examples
    vec.sort_by(|a, b| b.cmp(a)); // not linted to avoid suggesting `Reverse(b)` which would borrow
    vec.sort_by(|a, b| (b + 5).abs().cmp(&(a + 5).abs()));
    //~^ unnecessary_sort_by
    vec.sort_unstable_by(|a, b| id(-b).cmp(&id(-a)));
    //~^ unnecessary_sort_by
    // Negative examples (shouldn't be changed)
    let c = &7;
    vec.sort_by(|a, b| (b - a).cmp(&(a - b)));
    vec.sort_by(|_, b| b.cmp(&5));
    vec.sort_by(|_, b| b.cmp(c));
    vec.sort_unstable_by(|a, _| a.cmp(c));

    // Vectors of references are fine as long as the resulting key does not borrow
    let mut vec: Vec<&&&isize> = vec![&&&3, &&&6, &&&1, &&&2, &&&5];
    vec.sort_by(|a, b| (***a).abs().cmp(&(***b).abs()));
    //~^ unnecessary_sort_by
    vec.sort_unstable_by(|a, b| (***a).abs().cmp(&(***b).abs()));
    //~^ unnecessary_sort_by
    // `Reverse(b)` would borrow in the following cases, don't lint
    vec.sort_by(|a, b| b.cmp(a));
    vec.sort_unstable_by(|a, b| b.cmp(a));

    // No warning if element does not implement `Ord`
    let mut vec: Vec<Ref<usize>> = Vec::new();
    vec.sort_unstable_by(|a, b| a.cmp(b));
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

// The closure parameter is not dereferenced anymore, so non-Copy types can be linted
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
        //~^ unnecessary_sort_by
        args.sort_unstable_by(|a, b| a.name().cmp(&b.name()));
        //~^ unnecessary_sort_by
        // Reverse
        args.sort_by(|a, b| b.name().cmp(&a.name()));
        //~^ unnecessary_sort_by
        args.sort_unstable_by(|a, b| b.name().cmp(&a.name()));
        //~^ unnecessary_sort_by
    }
}

fn main() {
    unnecessary_sort_by();
    issue_5754::test();
    issue_6001::test();
}
