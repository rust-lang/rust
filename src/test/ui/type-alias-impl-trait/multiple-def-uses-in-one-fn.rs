// https://github.com/rust-lang/rust/issues/73481
// This test used to cause unsoundness, since one of the two possible
// resolutions was chosen at random instead of erroring due to conflicts.

#![feature(min_type_alias_impl_trait)]

type X<A, B> = impl Into<&'static A>;

fn f<A, B: 'static>(a: &'static A, b: B) -> (X<A, B>, X<B, A>) {
    //~^ ERROR defining use generics `[B, A]` differ from previous defining use
    //~| ERROR defining use generics `[B, A]` differ from previous defining use
    (a, a)
}

fn main() {
    println!("{}", <X<_, _> as Into<&String>>::into(f(&[1isize, 2, 3], String::new()).1));
}
