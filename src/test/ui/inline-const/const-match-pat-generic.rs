#![allow(incomplete_features)]
#![feature(inline_const_pat)]

// rust-lang/rust#82518: ICE with inline-const in match referencing const-generic parameter

fn foo<const V: usize>() {
  match 0 {
    const { V } => {},
    //~^ ERROR const parameters cannot be referenced in patterns [E0158]
    _ => {},
  }
}

fn main() {
    foo::<1>();
}
