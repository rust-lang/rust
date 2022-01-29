fn main() {
    let v: Vec(&str) = vec!['1', '2'];
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| ERROR mismatched types
}
