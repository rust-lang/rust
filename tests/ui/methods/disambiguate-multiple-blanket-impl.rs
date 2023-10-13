trait A {
    fn foo(&self);
}

trait B {
    fn foo(&self);
}

#[derive(Debug)]
struct S;

impl<T: std::fmt::Debug> A for T {
    fn foo(&self) {} //~ NOTE candidate #1
}

impl<T: std::fmt::Debug> B for T {
    fn foo(&self) {} //~ NOTE candidate #2
}

fn main() {
    let s = S;
    S::foo(&s); //~ ERROR multiple applicable items in scope
    //~^ NOTE multiple `foo` found
    //~| HELP disambiguate
    //~| HELP disambiguate
}

