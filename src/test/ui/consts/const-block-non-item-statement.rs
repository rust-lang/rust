enum Foo {
    Bar = { let x = 1; 3 }
    //~^ ERROR let bindings in constants are unstable
    //~| ERROR statements in constants are unstable
    //~| ERROR let bindings in constants are unstable
    //~| ERROR statements in constants are unstable
}

pub fn main() {}
