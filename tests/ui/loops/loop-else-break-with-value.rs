fn main() {
    let Some(1) = loop {
        //~^ NOTE `else` is attached to this loop
        //~| ERROR refutable pattern in local binding
        //~| NOTE not covered
        //~| NOTE for more information
        //~| NOTE matched value is of type
        //~| NOTE require an "irrefutable pattern"
        break Some(1)
    } else {
        //~^ ERROR `loop...else` loops are not supported
        //~| NOTE consider moving this `else` clause to a separate `if` statement and use a `bool` variable to control if it should run
        return;
    };
}
