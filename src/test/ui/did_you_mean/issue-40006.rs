impl dyn X { //~ ERROR cannot be made into an object
//~^ ERROR missing
    Y
}

struct S;

trait X { //~ ERROR missing
    X() {}
    fn xxx() { ### } //~ ERROR missing
    //~^ ERROR expected
    L = M; //~ ERROR missing
    Z = { 2 + 3 }; //~ ERROR expected one of
    ::Y (); //~ ERROR expected one of
}

impl S {
    pub hello_method(&self) { //~ ERROR missing
        println!("Hello");
    }
}

fn main() {
    S.hello_method();
}
