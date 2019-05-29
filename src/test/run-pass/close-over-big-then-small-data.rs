#![allow(dead_code)]
// If we use GEPi rather than GEP_tup_like when
// storing closure data (as we used to do), the u64 would
// overwrite the u16.

#![feature(box_syntax)]

struct Pair<A,B> {
    a: A, b: B
}

struct Invoker<A> {
    a: A,
    b: u16,
}

trait Invokable<A> {
    fn f(&self) -> (A, u16);
}

impl<A:Clone> Invokable<A> for Invoker<A> {
    fn f(&self) -> (A, u16) {
        (self.a.clone(), self.b)
    }
}

fn f<A:Clone + 'static>(a: A, b: u16) -> Box<dyn Invokable<A>+'static> {
    box Invoker {
        a: a,
        b: b,
    } as (Box<dyn Invokable<A>+'static>)
}

pub fn main() {
    let (a, b) = f(22_u64, 44u16).f();
    println!("a={} b={}", a, b);
    assert_eq!(a, 22u64);
    assert_eq!(b, 44u16);
}
