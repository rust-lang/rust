// if we use lhs == rhs in a macro, we should not suggest to swap the equality
// because the origin span of lhs and rhs can not be found. See issue #139050

use std::fmt::Debug;

pub fn foo<I: Iterator>(mut iter: I, value: &I::Item)
where
    Item: Eq + Debug, //~ ERROR cannot find type `Item` in this scope [E0412]
{
    debug_assert_eq!(iter.next(), Some(value)); //~ ERROR  mismatched types [E0308]
    assert_eq!(iter.next(), Some(value)); //~ ERROR  mismatched types [E0308]
}
fn main() {}
