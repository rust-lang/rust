fn main() {
    &{[1, 2, 3][4]};
    //~^ ERROR index out of bounds
    //~| ERROR reaching this expression at runtime will panic or abort
    //~| ERROR this expression will panic at runtime
}
