mod m {
//~^ ERROR `main` function not found
    // An inferred main entry point
    // must appear at the top of the crate
    fn main() { }
}
