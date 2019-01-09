struct A { foo: isize }

fn a() -> A { panic!() }

fn main() {
    let A { , } = a(); //~ ERROR expected ident
                       //~| ERROR pattern does not mention field `foo`
}
