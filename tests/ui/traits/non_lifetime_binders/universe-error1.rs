// regression test for #109505
#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

trait Other<U: ?Sized> {}

impl<U: ?Sized> Other<U> for U {}

//  - o: `for<T> T: Trait<?0>`
//    - c: `for<U> U: Trait<U>`
//      - U = ?1
//      - T = ?1
//      - ?0 = ?1

#[rustfmt::skip]
fn foo<U: ?Sized>()
where
    for<T> T: Other<U> {}

fn bar() {
    foo::<_>(); //~ ERROR the trait bound `T: Other<_>` is not satisfied
}

fn main() {}
