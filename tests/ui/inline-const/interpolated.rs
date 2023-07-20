// check-pass

#![feature(inline_const)]

// This used to be unsupported since the parser first tries to check if we have
// any nested items, and then checks for statements (and expressions). The heuristic
// that we were using to detect the beginning of a const item was incorrect, so
// this used to fail.
macro_rules! m {
    ($b:block) => {
        fn foo() {
            const $b
        }
    }
}

// This has worked since inline-consts were implemented, since the position that
// the const block is located at doesn't support nested items (e.g. because
// `let x = const X: u32 = 1;` is invalid), so there's no ambiguity parsing the
// inline const.
macro_rules! m2 {
    ($b:block) => {
        fn foo2() {
            let _ = const $b;
        }
    }
}

m!({});
m2!({});

fn main() {}
