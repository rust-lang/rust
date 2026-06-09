impl dyn A {
    Y
} //~ ERROR expected one of `!` or `::`, found `}`

struct S;

trait X {
    X() {} //~ ERROR missing `fn` for function definition
    fn xxx() { ### }
    L = M;
    Z = { 2 + 3 };
    ::Y ();
}

trait A {
    X() {} //~ ERROR missing `fn` for function definition
}
trait B {
    fn xxx() { ### } //~ ERROR expected
}
trait C {
    L = M; //~ ERROR expected one of `!` or `::`, found `=`
}
trait D {
    Z = { 2 + 3 }; //~ ERROR expected one of `!` or `::`, found `=`
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
    S.hello_method(); //~ ERROR no method named `hello_method` found
}
