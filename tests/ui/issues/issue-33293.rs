fn main() {
    match 0 {
        aaa::bbb(_) => ()
        //~^ ERROR failed to resolve: use of undeclared crate or module `aaa`
    };
}
