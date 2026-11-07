fn main () {
    struct Foo { a: bool };

    let f = Foo(); //~ ERROR E0423
}

fn bar() {
    struct S { x: i32, y: i32 }
    #[derive(PartialEq)]
    struct T {}

    if let S { x: _x, y: 2 } = S { x: 1, y: 2 } { println!("Ok"); }
    //~^ ERROR struct literals are not allowed here
    if T {} == T {} { println!("Ok"); }
    //~^ ERROR E0423
    //~| ERROR expected expression, found `==`
}

fn foo() {
    for _ in std::ops::Range { start: 0, end: 10 } {}
    //~^ ERROR struct literals are not allowed here
}
