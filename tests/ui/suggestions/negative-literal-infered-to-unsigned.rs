fn main() {
    for x in -5..5 {
        //~^ ERROR: the trait bound `usize: Neg` is not satisfied
        //~| HELP: consider specifying an integer type that can be negative
        do_something(x);
    }
    let x = -5;
    //~^ ERROR: the trait bound `usize: Neg` is not satisfied
    //~| HELP: consider specifying an integer type that can be negative
    do_something(x);
}

fn do_something(_val: usize) {}
