fn main() {
    if let Some(b) = None {
        //~^ NOTE `if` and `else` have incompatible types
        ()
        //~^ NOTE expected because of this
    } else {
        1
    };
    //~^^ ERROR: `if` and `else` have incompatible types
    //~| NOTE expected `()`, found integer
}
