#[allow(unused)]
fn foo() {
    //~^ NOTE possibly return type missing here?
    vec!['a'].iter().map(|c| c)
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected `()`, found struct `Map`
    //~| NOTE expected unit type `()`
}

fn main() {}
