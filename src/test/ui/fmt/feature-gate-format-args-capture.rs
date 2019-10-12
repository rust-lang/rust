fn main() {
    format!("{foo}");                //~ ERROR: there is no argument named `foo`

    // panic! doesn't hit format_args! unless there are two or more arguments.
    panic!("{foo} {bar}", bar=1);    //~ ERROR: there is no argument named `foo`
}
