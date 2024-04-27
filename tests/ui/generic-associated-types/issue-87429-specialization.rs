//@ check-fail

#![feature(specialization)]
//~^ WARN incomplete

trait Family {
    type Member<'a>: for<'b> PartialEq<Self::Member<'b>>;
}

struct I32Family;

impl Family for I32Family {
    default type Member<'a> = i32;
}

struct Foo;
struct FooFamily;

impl Family for FooFamily {
    default type Member<'a> = Foo;
    //~^ ERROR can't compare
}

fn main() {}
