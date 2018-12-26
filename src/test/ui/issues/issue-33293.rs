fn main() {
    match 0 {
        aaa::bbb(_) => ()
        //~^ ERROR failed to resolve: use of undeclared type or module `aaa`
    };
}
