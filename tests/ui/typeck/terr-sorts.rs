//@ dont-require-annotations: NOTE

struct Foo {
    a: isize,
    b: isize,
}

type Bar = Box<Foo>;

fn want_foo(f: Foo) {}
fn have_bar(b: Bar) {
    want_foo(b); //~  ERROR mismatched types
                 //~| NOTE expected struct `Foo`
                 //~| NOTE found struct `Box<Foo>`
}

fn main() {}
