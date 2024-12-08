#!//bin/bash


// This could not possibly be a shebang & also a valid rust file, since a Rust file
// can't start with `[`
/*
    [ (mixing comments to also test that we ignore both types of comments)

 */

[allow(unused_variables)]

//@ check-pass
//@ reference: input.shebang.inner-attribute
fn main() {
    let x = 5;
}
