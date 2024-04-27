impl dyn A {
    Y
} //~ ERROR expected one of `!` or `::`, found `}`

struct S;

trait X {
    X() {} //~ ERROR expected one of `!` or `::`, found `(`
    fn xxx() { ### }
    L = M;
    Z = { 2 + 3 };
    ::Y ();
}

trait A {
    X() {} //~ ERROR expected one of `!` or `::`, found `(`
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
    S.hello_method(); //~ no method named `hello_method` found
}
