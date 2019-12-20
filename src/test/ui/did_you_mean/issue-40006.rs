impl dyn A { //~ ERROR missing
    Y
}

struct S;

trait X { //~ ERROR missing
    X() {}
    fn xxx() { ### }
    L = M;
    Z = { 2 + 3 };
    ::Y ();
}

trait A { //~ ERROR missing
    X() {}
}
trait B {
    fn xxx() { ### } //~ ERROR expected
}
trait C { //~ ERROR missing `fn`, `type`, or `const` for associated-item declaration
    L = M;
}
trait D { //~ ERROR missing `fn`, `type`, or `const` for associated-item declaration
    Z = { 2 + 3 };
}
trait E {
    ::Y (); //~ ERROR expected one of
}

impl S {
    pub hello_method(&self) { //~ ERROR missing
        println!("Hello");
    }
}

fn main() {
    S.hello_method(); //~ no method named `hello_method` found for type `S` in the current scope
}
