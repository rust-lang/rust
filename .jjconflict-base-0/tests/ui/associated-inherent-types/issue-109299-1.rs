#![feature(inherent_associated_types, non_lifetime_binders, type_alias_impl_trait)]
#![allow(incomplete_features)]

struct Lexer<T>(T);

impl Lexer<i32> {
    type Cursor = ();
}

type X = impl for<T> Fn() -> Lexer<T>::Cursor;
//~^ ERROR: unconstrained opaque type

fn main() {}
