//@ check-pass
#![feature(type_alias_impl_trait)]

type X<A: ToString + Clone, B: ToString + Clone> = impl ToString;

#[define_opaque(X)]
fn f<A: ToString + Clone, B: ToString + Clone>(a: A, b: B) -> (X<A, B>, X<A, B>) {
    (a.clone(), a)
}

type Tait<'x> = impl Sized;
#[define_opaque(Tait)]
fn define<'a: 'b, 'b: 'a>(x: &'a u8, y: &'b u8) -> (Tait<'a>, Tait<'b>) {
    ((), ())
}

fn main() {
    println!("{}", <X<_, _> as ToString>::to_string(&f(42_i32, String::new()).1));
}
