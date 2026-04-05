// issue: <https://github.com/rust-lang/rust/issues/152338>

use std::iter::Iterator;

struct Ty<T>
//~^ ERROR type parameter `T` is never used
where
    T: Iterator,
    <T>::Item = i32, {}
//~^ ERROR equality constraints are not yet supported in `where` clauses

fn main() {}
