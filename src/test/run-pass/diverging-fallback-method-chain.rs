#![allow(unused_imports)]
// Test a regression found when building compiler. The `produce()`
// error type `T` winds up getting unified with result of `x.parse()`;
// the type of the closure given to `unwrap_or_else` needs to be
// inferred to `usize`.

use std::num::ParseIntError;

fn produce<T>() -> Result<&'static str, T> {
    Ok("22")
}

fn main() {
    let x: usize = produce()
        .and_then(|x| x.parse())
        .unwrap_or_else(|_| panic!());
    println!("{}", x);
}
