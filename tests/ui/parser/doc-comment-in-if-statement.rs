fn main() {
    if true /*!*/ {}
    //~^ ERROR outer attributes are not allowed on
    //~| ERROR expected outer doc comment
}
