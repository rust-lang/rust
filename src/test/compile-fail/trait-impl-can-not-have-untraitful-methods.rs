trait A { }

impl int: A {
    fn foo() { } //~ ERROR method `foo` is not a member of trait `A`
}

fn main() { }