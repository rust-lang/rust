fn main() {
    match 0 {
        aaa::bbb(_) => ()
        //~^ ERROR failed to resolve. Use of undeclared type or module `aaa`
    };
}
