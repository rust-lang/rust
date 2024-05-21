enum E {
    Foo,
    Bar(String)
}

struct S {
    x: E
}

fn f(x: String) {}

fn main() {
    let s = S { x: E::Bar("hello".to_string()) };
    match &s.x { //~ ERROR cannot move
        &E::Foo => {}
        &E::Bar(identifier) => f(identifier.clone())
    };
    match &s.x {
        &E::Foo => {}
        &E::Bar(ref identifier) => println!("{}", *identifier)
    };
    if let &E::Bar(identifier) = &s.x { //~ ERROR cannot move
        f(identifier.clone());
    };
    let &E::Bar(identifier) = &s.x else { //~ ERROR cannot move
        return;
    };
    f(identifier.clone());
}
