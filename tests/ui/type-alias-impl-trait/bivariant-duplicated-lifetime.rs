//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(unconditional_recursion)]

type Opaque<'a> = impl Sized + 'a;

fn test<'a>() -> Opaque<'a> {
    let _: () = test::<'a>();
}

fn main() {}
