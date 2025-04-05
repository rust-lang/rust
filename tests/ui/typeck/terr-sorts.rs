struct Foo {
    a: isize,
    b: isize,
}

type Bar = Box<Foo>;

fn want_foo(f: Foo) {}
fn have_bar(b: Bar) {
    want_foo(b); //~  ERROR mismatched types
                 //~| NOTE_NONVIRAL expected struct `Foo`
                 //~| NOTE_NONVIRAL found struct `Box<Foo>`
}

fn main() {}
