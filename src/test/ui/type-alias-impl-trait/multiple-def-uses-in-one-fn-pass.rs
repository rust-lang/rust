// check-pass
#![feature(min_type_alias_impl_trait)]

type X<A: ToString + Clone, B: ToString + Clone> = impl ToString;

fn f<A: ToString + Clone, B: ToString + Clone>(a: A, b: B) -> (X<A, B>, X<A, B>) {
    (a.clone(), a)
}

fn main() {
    println!("{}", <X<_, _> as ToString>::to_string(&f(42_i32, String::new()).1));
}
