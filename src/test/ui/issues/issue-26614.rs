// compile-pass
// skip-codegen
#![allow(warnings)]
trait Mirror {
    type It;
}

impl<T> Mirror for T {
    type It = Self;
}



fn main() {
    let c: <u32 as Mirror>::It = 5;
    const CCCC: <u32 as Mirror>::It = 5;
}
