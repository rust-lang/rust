// We allow for literals to implicitly be anon consts still regardless
// of whether a const block is placed around them or not
//
// However, we don't allow so for const arguments in field init positions.
// This is just harder to implement so we did not do so.

#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

trait Trait {
    #[type_const]
    const ASSOC: isize;
}

fn ace<T: Trait<ASSOC = 1, ASSOC = -1>>() {}
fn repeat_count() {
    [(); 1];
}
type ArrLen = [(); 1];
struct Foo<const N: isize>;
type NormalArg = (Foo<1>, Foo<-1>);

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
struct ADT { field: u8 }

fn struct_expr() {
    fn takes_n<const N: ADT>() {}

    takes_n::<{ ADT { field: 1 } }>();
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
    takes_n::<{ ADT { field: const { 1 } } }>();
}

fn main() {}
