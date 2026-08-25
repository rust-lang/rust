//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

const fn foo() {}
const fn bar() {}
fn baz() {}

const fn caller(branch: bool) {
    let mut x = if branch {
      foo
    } else {
      bar
    };
    x = baz;
}

fn main() {}
