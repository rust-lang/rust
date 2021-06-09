// https://github.com/rust-lang/rust/issues/73481
// This test used to cause unsoundness, since one of the two possible
// resolutions was chosen at random instead of erroring due to conflicts.

#![feature(min_type_alias_impl_trait)]

type X<A: ToString + Clone, B: ToString + Clone> = impl ToString;
//~^ ERROR could not find defining uses

fn f<A: ToString + Clone, B: ToString + Clone>(a: A, b: B) -> (X<A, B>, X<B, A>) {
    (a.clone(), a)
}

fn main() {
    println!("{}", <X<_, _> as ToString>::to_string(&f(42_i32, String::new()).1));
}
