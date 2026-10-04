#!//bin/bash

//@ check-pass
//@ reference: shebang.syntax
fn main() {
    println!("a valid shebang (that is also a rust comment)")
}
