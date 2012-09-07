fn reproduce<T:Copy>(t: T) -> fn@() -> T {
    fn@() -> T { t }
}

fn main() {
    // type of x is the variable X,
    // with the lower bound @mut int
    let x = @mut 3;

    // type of r is fn@() -> X
    let r = reproduce(x);

    // Requires that X be a subtype of
    // @mut int.
    let f: @mut int = r();

    // OK.
    let g: @const int = r();

    // Bad.
    let h: @int = r(); //~ ERROR (values differ in mutability)
}
