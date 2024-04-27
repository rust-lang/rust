// Test that anonymous lifetimes are not permitted in struct declarations

struct Foo {
    x: &isize //~ ERROR missing lifetime specifier
}

fn main() {}
