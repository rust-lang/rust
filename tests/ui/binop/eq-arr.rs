fn main() {
    struct X;
    //~^ HELP: consider annotating `X` with `#[derive(PartialEq)]`
    let xs = [X, X, X];
    let eq = xs == [X, X, X];
    //~^ ERROR: can't compare `[X; 3]`
}
