fn main() {
    let v: Vec(&str) = vec!['1', '2'];
    //~^ ERROR parenthesized parameters may only be used with a trait
}
