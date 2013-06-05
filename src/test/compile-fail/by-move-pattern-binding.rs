enum E {
    Foo,
    Bar(~str)
}

struct S {
    x: E
}

fn f(x: ~str) {}

fn main() {
    let s = S { x: Bar(~"hello") };
    match &s.x {
        &Foo => {}
        &Bar(identifier) => f(copy identifier)  //~ ERROR cannot move
    };
    match &s.x {
        &Foo => {}
        &Bar(ref identifier) => println(*identifier)
    };
}
