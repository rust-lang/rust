#!//bin/bash

//@ check-pass
//@ reference: input.shebang
fn main() {
    println!("a valid shebang (that is also a rust comment)")
}
