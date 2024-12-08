// Test that we correctly infer variance for type parameters in
// various types and traits.

#![feature(rustc_attrs)]

#[rustc_variance]
struct TestImm<A, B> { //~ ERROR [A: +, B: +]
    x: A,
    y: B,
}

#[rustc_variance]
struct TestMut<A, B:'static> { //~ ERROR [A: +, B: o]
    x: A,
    y: &'static mut B,
}

#[rustc_variance]
struct TestIndirect<A:'static, B:'static> { //~ ERROR [A: +, B: o]
    m: TestMut<A, B>
}

#[rustc_variance]
struct TestIndirect2<A:'static, B:'static> { //~ ERROR [A: o, B: o]
    n: TestMut<A, B>,
    m: TestMut<B, A>
}

trait Getter<A> {
    fn get(&self) -> A;
}

trait Setter<A> {
    fn set(&mut self, a: A);
}

#[rustc_variance]
struct TestObject<A, R> { //~ ERROR [A: o, R: o]
    n: Box<dyn Setter<A>+Send>,
    m: Box<dyn Getter<R>+Send>,
}

fn main() {}
