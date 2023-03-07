#![feature(concat_idents)]

fn main() {
    let x = concat_idents!(); //~ ERROR concat_idents! takes 1 or more arguments
}
