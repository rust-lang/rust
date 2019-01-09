fn main() {
    env!("__HOPEFULLY_NOT_DEFINED__");
    //~^ ERROR: environment variable `__HOPEFULLY_NOT_DEFINED__` not defined
}
