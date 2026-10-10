//@ run-rustfix

use std::sync::Arc;

fn main() {
    let _ = 7u32 as Option<_>;
    //~^ ERROR non-primitive cast: `u32` as `Option<_>`
    let _ = "String" as Arc<str>;
    //~^ ERROR non-primitive cast: `&'static str` as `Arc<str>`
}
