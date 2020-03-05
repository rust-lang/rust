fn main() {
    if true /*!*/ {}
    //~^ ERROR expected `{`, found doc comment `/*!*/`
    //~| ERROR expected outer doc comment
}
