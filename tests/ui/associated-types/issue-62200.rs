struct S {}

trait T<'a> {
    type A;
}

impl T<'_> for S {
    type A = u32;
}

fn foo(x: impl Fn(<S as T<'_>>::A) -> <S as T<'_>>::A) {}
//~^ ERROR binding for associated type `Output` references an anonymous lifetime
//~| NOTE lifetimes appearing in an associated or opaque type are not considered constrained
//~| NOTE consider introducing a named lifetime parameter

fn main() {}
