fn main() {
    for _ in 0..1 {

    } else {
        //~^ ERROR `for...else` loops are not supported
        //~| NOTE consider moving this `else` clause to a separate `if` statement and use a `bool` variable to control if it should run
    }
}
