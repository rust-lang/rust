// compile-flags: -Z parse-only

struct A { foo: isize }

fn a() -> A { panic!() }

fn main() {
    let A { , } = a(); //~ ERROR: expected ident
}
