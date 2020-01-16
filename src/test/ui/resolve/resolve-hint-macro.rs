fn main() {
    assert(true);
    //~^ ERROR expected function, found macro `assert`
    format(""); // shouldn't suggest import candidates
    //~^ ERROR expected function, found macro `format`
}
