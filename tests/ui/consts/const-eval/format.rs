const fn failure() {
    panic!("{:?}", 0);
    //~^ ERROR cannot call non-const formatting macro in constant functions
}

const fn print() {
    println!("{:?}", 0);
    //~^ ERROR cannot call non-const formatting macro in constant functions
    //~| ERROR cannot call non-const function `_print` in constant functions
}

const fn format_args() {
    format_args!("{}", 0);
    //~^ ERROR cannot call non-const formatting macro in constant functions
}

fn main() {}
