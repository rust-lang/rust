struct Foo {
    a: isize,
    b: isize,
}

type Bar = Box<Foo>;

fn want_foo(f: Foo) {}
fn have_bar(b: Bar) {
    want_foo(b); //~  ERROR mismatched types
                 //~| expected struct `Foo`
                 //~| found struct `std::boxed::Box<Foo>`
}

fn main() {}
