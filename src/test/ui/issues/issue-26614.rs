#![feature(rustc_attrs)]
#![allow(warnings)]

trait Mirror {
    type It;
}

impl<T> Mirror for T {
    type It = Self;
}


#[rustc_error]
fn main() { //~ ERROR compilation successful
    let c: <u32 as Mirror>::It = 5;
    const CCCC: <u32 as Mirror>::It = 5;
}
