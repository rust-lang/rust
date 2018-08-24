struct foo {
    a: isize,
    b: isize,
}

struct bar {
    a: isize,
    b: usize,
}

fn want_foo(f: foo) {}
fn have_bar(b: bar) {
    want_foo(b); //~  ERROR mismatched types
                 //~| expected type `foo`
                 //~| found type `bar`
                 //~| expected struct `foo`, found struct `bar`
}

fn main() {}
