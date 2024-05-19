//@ check-pass

// From https://github.com/rust-lang/rust/issues/72476
// and https://github.com/rust-lang/rust/issues/89393

trait Trait {
    type Projection;
}

struct A;
impl Trait for A {
    type Projection = bool;
}

struct B;
impl Trait for B {
    type Projection = (u32, u32);
}

struct Next<T: Trait>(T::Projection);

fn foo1(item: Next<A>) {
    match item {
        Next(true) => {}
        Next(false) => {}
    }
}

fn foo2(x: <A as Trait>::Projection) {
    match x {
        true => {}
        false => {}
    }
}

fn foo3(x: Next<B>) {
    let Next((_, _)) = x;
    match x {
        Next((_, _)) => {}
    }
}

fn foo4(x: <B as Trait>::Projection) {
    let (_, _) = x;
    match x {
        (_, _) => {}
    }
}

fn foo5<T: Trait>(x: <T as Trait>::Projection) {
    match x {
        _ => {}
    }
}

fn main() {}
