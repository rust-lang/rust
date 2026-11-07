// Test that anonymous lifetimes are not permitted in enum declarations

enum Foo {
    Bar(&isize) //~ ERROR missing lifetime specifier
}

fn main() {}
