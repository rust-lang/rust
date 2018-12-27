// Copyright 2014–2017 The Rust Project Developers. See the COPYRIGHT
// http://rust-lang.org/COPYRIGHT.
//

// compile-pass

// this tests the `unknown_lint` lint, especially the suggestions

// the suggestion only appears if a lint with the lowercase name exists
#[allow(FOO_BAR)]
// the suggestion appears on all-uppercase names
#[warn(DEAD_CODE)]
// the suggestion appears also on mixed-case names
#[deny(Warnings)]
fn main() {
    unimplemented!();
}
