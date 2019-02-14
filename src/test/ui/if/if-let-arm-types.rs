fn main() {
    if let Some(b) = None {
        //~^ NOTE if let` arms have incompatible types
        ()
    } else {
        1
    };
    //~^^ ERROR: `if let` arms have incompatible types
    //~| NOTE expected (), found integer
    //~| NOTE expected type `()`
}
