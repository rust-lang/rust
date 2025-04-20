#![feature(concat_idents)]
#![expect(deprecated)] // concat_idents is deprecated

fn main() {
    let x = concat_idents!(); //~ ERROR `concat_idents!()` takes 1 or more arguments
}
