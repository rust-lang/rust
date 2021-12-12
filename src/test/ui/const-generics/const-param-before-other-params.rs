fn bar<const X: u8, 'a>(_: &'a ()) {
    //~^ ERROR lifetime parameters must be declared prior to const parameters
}

fn foo<const X: u8, T>(_: &T) {}

fn main() {}
