struct foo {
    a: isize,
    b: isize,
}

type bar = Box<foo>;

fn want_foo(f: foo) {}
fn have_bar(b: bar) {
    want_foo(b); //~  ERROR mismatched types
                 //~| expected type `foo`
                 //~| found type `std::boxed::Box<foo>`
}

fn main() {}
