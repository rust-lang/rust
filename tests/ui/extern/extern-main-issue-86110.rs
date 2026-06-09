// missing and missing2 exist to make sure that the error only happens on a `main` declaration
extern "C" {
    fn missing();
    fn main();
    //~^ ERROR the `main` function cannot be declared in an `extern` block
    fn missing2();
}
