#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

struct ConstAssert<const COND: bool>;
trait True {}
impl True for ConstAssert<true> {}

struct Range<T: PartialOrd, const MIN: T, const MAX: T>(T)
//~^ ERROR the type of const parameters must not depend on other generic parameters
//~| ERROR the type of const parameters must not depend on other generic parameters
where
    ConstAssert<{ MIN <= MAX }>: True;

fn main() {}
