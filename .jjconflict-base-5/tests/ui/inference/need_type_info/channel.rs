// Test that we suggest specifying the generic argument of `channel`
// instead of the return type of that function, which is a lot more
// complex.
use std::sync::mpsc::channel;

fn no_tuple() {
    let _data =
        channel(); //~ ERROR type annotations needed
}

fn tuple() {
    let (_sender, _receiver) =
        channel(); //~ ERROR type annotations needed
}

fn main() {
    no_tuple();
    tuple();
}
