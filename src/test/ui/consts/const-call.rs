fn f(x: usize) -> usize {
    x
}

fn main() {
    let _ = [0; f(2)];
    //~^ ERROR calls in constants are limited to constant functions
    //~| ERROR evaluation of constant value failed
}
