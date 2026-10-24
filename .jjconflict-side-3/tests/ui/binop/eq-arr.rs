fn main() {
    struct X;
    //~^ HELP consider annotating `X` with `#[derive(PartialEq)]`
    let xs = [X, X, X];
    let eq = xs == [X, X, X];
    //~^ ERROR binary operation `==` cannot be applied to type `[X; 3]`
}
