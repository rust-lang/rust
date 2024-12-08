fn main() {
    env!("CARGO__HOPEFULLY_NOT_DEFINED__");
    //~^ ERROR: environment variable `CARGO__HOPEFULLY_NOT_DEFINED__` not defined
}
