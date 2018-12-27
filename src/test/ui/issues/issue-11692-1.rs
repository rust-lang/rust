fn main() {
    print!(testo!());
    //~^ ERROR: format argument must be a string literal
    //~| ERROR: cannot find macro `testo!` in this scope
}
