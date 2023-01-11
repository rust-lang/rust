// run-pass

#![allow(non_camel_case_types)]
#![allow(dead_code)]

struct pair<A,B> {
    a: A, b: B
}

trait Invokable<A> {
    fn f(&self) -> (A, u16);
}

struct Invoker<A> {
    a: A,
    b: u16,
}

impl<A:Clone> Invokable<A> for Invoker<A> {
    fn f(&self) -> (A, u16) {
        (self.a.clone(), self.b)
    }
}

fn f<A:Clone + 'static>(a: A, b: u16) -> Box<dyn Invokable<A>+'static> {
    Box::new(Invoker {
        a: a,
        b: b,
    }) as Box<dyn Invokable<A>+'static>
}

pub fn main() {
    let (a, b) = f(22_u64, 44u16).f();
    println!("a={} b={}", a, b);
    assert_eq!(a, 22u64);
    assert_eq!(b, 44u16);
}
