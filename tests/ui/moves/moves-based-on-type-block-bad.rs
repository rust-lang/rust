#![feature(deref_patterns)]


struct S {
    x: Box<E>
}

enum E {
    Foo(Box<S>),
    Bar(Box<isize>),
    Baz
}

fn f<G>(s: &S, g: G) where G: FnOnce(&S) {
    g(s)
}

fn main() {
    let s = S { x: Box::new(E::Bar(Box::new(42))) };
    loop {
        f(&s, |hellothere| {
            match hellothere.x { //~ ERROR cannot move out
                E::Foo(_) => {}
                E::Bar(x) => println!("{}", x.to_string()),
                E::Baz => {}
            }
        })
    }
}
