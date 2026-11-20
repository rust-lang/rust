#![allow(unused_parens)]
fn main() {
    {
        (break); //~ ERROR `break` outside of a loop or labeled block
    };
    {
        ((break)); //~ ERROR `break` outside of a loop or labeled block
    };
}
