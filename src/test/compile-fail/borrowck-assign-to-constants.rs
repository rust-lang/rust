const foo: int = 5;

fn main() {
    // assigning to various global constants
    None = Some(3); //~ ERROR assigning to static item
    foo = 6; //~ ERROR assigning to static item
}