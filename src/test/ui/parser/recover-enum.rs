fn main() {
    enum Test {
        Very
        //~^ ERROR missing comma
        Bad(usize)
        //~^ ERROR missing comma
        Stuff { a: usize }
        //~^ ERROR missing comma
        Here
    }
}
