trait A {
    fn foo(&self);
}

trait B {
    fn foo(&self);
}

struct S;

impl A for S {
    fn foo(&self) {} //~ NOTE candidate #1
}

impl B for S {
    fn foo(&self) {} //~ NOTE candidate #2
}

fn main() {
    let s = S;
    S::foo(&s); //~ ERROR multiple applicable items in scope
    //~^ NOTE multiple `foo` found
    //~| HELP disambiguate
    //~| HELP disambiguate
}

