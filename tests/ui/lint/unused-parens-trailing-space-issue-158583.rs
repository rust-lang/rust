fn main() {
    #[deny(unused_parens)]
    let _x = (3 + 6);
    //~^ ERROR unnecessary parentheses around assigned value
}
