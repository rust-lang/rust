impl dyn A {
    Y //~ ERROR non-item in item list
}

struct S;

trait X {
    X() {} //~ ERROR non-item in item list
    fn xxx() { ### }
    L = M;
    Z = { 2 + 3 };
    ::Y ();
}

trait A {
    X() {} //~ ERROR non-item in item list
}
trait B {
    fn xxx() { ### } //~ ERROR expected
}
trait C {
    L = M; //~ ERROR non-item in item list
}
trait D {
    Z = { 2 + 3 }; //~ ERROR non-item in item list
}
trait E {
    ::Y (); //~ ERROR non-item in item list
}

impl S {
    pub hello_method(&self) { //~ ERROR missing
        println!("Hello");
    }
}

fn main() {
    S.hello_method(); //~ no method named `hello_method` found
}
