enum E {
    Foo(String, String, String),
}

struct Bar {
    a: String,
    b: String,
}

fn main() {
    let bar = Bar { a: "1".to_string(), b: "2".to_string() };
    match E::Foo("".into(), "".into(), "".into()) {
        E::Foo(a, b, ref c) => {}
//~^ ERROR cannot bind by-move and by-ref in the same pattern
    }
    match bar {
        Bar {a, ref b} => {}
//~^ ERROR cannot bind by-move and by-ref in the same pattern
    }
}
