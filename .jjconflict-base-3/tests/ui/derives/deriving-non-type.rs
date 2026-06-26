#![allow(dead_code)]

struct S;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to `struct`s, `enum`s and `union`s
trait T { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to `struct`s, `enum`s and `union`s
impl S { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to `struct`s, `enum`s and `union`s
impl T for S { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to `struct`s, `enum`s and `union`s
static s: usize = 0;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to `struct`s, `enum`s and `union`s
const c: usize = 0;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to `struct`s, `enum`s and `union`s
mod m { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to `struct`s, `enum`s and `union`s
extern "C" { }

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to `struct`s, `enum`s and `union`s
type A = usize;

#[derive(PartialEq)] //~ ERROR: `derive` may only be applied to `struct`s, `enum`s and `union`s
fn main() { }
