fn main() {
    let Some(1) = loop {
        //~^ NOTE `else` is attached to this loop
        break Some(1)
    } else {
        //~^ ERROR `loop...else` loops are not supported
        //~| NOTE consider moving this `else` clause to a separate `if` statement and use a `bool` variable to control if it should run
        return;
    };
}
