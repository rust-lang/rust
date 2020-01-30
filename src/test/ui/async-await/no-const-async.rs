// edition:2018
// compile-flags: --crate-type lib

pub const async fn x() {}
//~^ ERROR functions cannot be both `const` and `async`
