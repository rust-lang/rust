//@ run-rustfix
fn wat<T>(t: &T) -> T {
    t.clone() //~ ERROR E0308
}

struct Foo;

fn wut(t: &Foo) -> Foo {
    t.clone() //~ ERROR E0308
}

fn main() {
    wat(&42);
    wut(&Foo);
}
