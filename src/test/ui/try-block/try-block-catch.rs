// compile-flags: --edition 2018

#![feature(try_blocks)]

fn main() {
    let res: Option<bool> = try {
        true
    } catch { };
    //~^ ERROR `try {} catch` is not a valid syntax
}
