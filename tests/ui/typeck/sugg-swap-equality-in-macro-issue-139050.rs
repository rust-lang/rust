// if we use lhs == rhs in a macro, we should not suggest to swap the equality
// because the origin span of lhs and rhs can not be found. See issue #139050

//@ aux-build:extern-macro-issue-139050.rs
//@ aux-crate:ext=extern-macro-issue-139050.rs

extern crate ext;

use std::fmt::Debug;

macro_rules! eq_local {
    (assert $a:expr, $b:expr) => {
        match (&$a, &$b) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    //~^ ERROR  mismatched types [E0308]
                    panic!(
                        "assertion failed: `(left == right)`\n  left: `{:?}`,\n right: `{:?}`",
                        left_val, right_val
                    );
                }
            }
        }
    };
}

pub fn foo<I: Iterator>(mut iter: I, value: &I::Item)
where
    Item: Eq + Debug, //~ ERROR cannot find type `Item` in this scope [E0425]
{
    ext::eq!(assert iter.next(), Some(value)); //~ ERROR  mismatched types [E0308]
    eq_local!(assert iter.next(), Some(value));
}
fn main() {}
