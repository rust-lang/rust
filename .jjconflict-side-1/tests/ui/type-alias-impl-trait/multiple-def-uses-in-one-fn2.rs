// https://github.com/rust-lang/rust/issues/73481
// This test used to cause unsoundness, since one of the two possible
// resolutions was chosen at random instead of erroring due to conflicts.

#![feature(type_alias_impl_trait)]

type X<A: ToString + Clone, B: ToString + Clone> = impl ToString;

#[define_opaque(X)]
fn f<A: ToString + Clone, B: ToString + Clone>(a: A, b: B) -> (X<A, B>, X<B, A>) {
    //~^ ERROR concrete type differs from previous defining opaque type
    (a.clone(), a)
}

fn main() {
    println!("{}", <X<_, _> as ToString>::to_string(&f(42_i32, String::new()).1));
}
