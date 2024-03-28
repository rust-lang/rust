//@ run-rustfix

#![allow(dead_code)]

trait Trait<T>
  where T: for<'a> Trait<T> + 'b { } //~ ERROR use of undeclared lifetime name `'b`

fn main() {}
