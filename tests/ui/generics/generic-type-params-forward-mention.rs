// Ensure that we get an error and not an ICE for this problematic case.
struct Foo<T = Option<U>, U = bool>(T, U);
//~^ ERROR generic parameter defaults cannot reference parameters before they are declared
fn main() {
    let x: Foo;
}
