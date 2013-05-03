struct S {
    x: ~E
}

enum E {
    Foo(~S),
    Bar(~int),
    Baz
}

fn f(s: &S, g: &fn(&S)) {
    g(s)
}

fn main() {
    let s = S { x: ~Bar(~42) };
    loop {
        do f(&s) |hellothere| {
            match hellothere.x {    //~ ERROR moving out of immutable field
                ~Foo(_) => {}
                ~Bar(x) => io::println(x.to_str()),
                ~Baz => {}
            }
        }
    }
}
