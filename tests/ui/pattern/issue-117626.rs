//@ check-pass

#[derive(PartialEq)]
struct NonMatchable;

impl Eq for NonMatchable {}

#[derive(PartialEq, Eq)]
enum Foo {
    A(NonMatchable),
    B(*const u8),
}

const CONST: Foo = Foo::B(std::ptr::null());

fn main() {
    match CONST {
        CONST => 0,
        _ => 1,
    };
}
