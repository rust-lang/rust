struct Foo {
    a: isize,
    b: isize,
}

type Bar = Box<Foo>;

fn want_foo(f: Foo) {}
fn have_bar(b: Bar) {
    want_foo(b); //~  ERROR arguments to this function are incorrect
                 //~| expected struct `Foo`
                 //~| found struct `Box<Foo>`
}

fn main() {}
