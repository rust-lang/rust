// check-fail

#![feature(associated_type_defaults)]
#![feature(generic_associated_types)]

trait Family {
    // Fine, i32: PartialEq<i32>
    type Member<'a>: for<'b> PartialEq<Self::Member<'b>> = i32;
}

struct Foo;
trait Family2 {
    // Not fine, not Foo: PartialEq<Foo>
    type Member<'a>: for<'b> PartialEq<Self::Member<'b>> = Foo;
    //~^ ERROR can't compare
}

fn main() {}
