fn main() {
    let _ = loop {
        //~^ NOTE `else` is attached to this loop
    } else {
        //~^ ERROR `loop...else` loops are not supported
        //~| NOTE consider moving this `else` clause to a separate `if` statement and use a `bool` variable to control if it should run
    };
}
