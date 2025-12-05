enum TestEnum {
    Works,
    /// Some documentation
    Self, //~ ERROR expected identifier, found keyword `Self`
    //~^ HELP enum variants can be
}
