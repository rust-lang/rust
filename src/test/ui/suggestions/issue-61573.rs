fn main() {
    let a, b = x;
    //~^ ERROR unexpected `,` in pattern
    //~| SUGGESTION try adding parentheses to match on a tuple
}
