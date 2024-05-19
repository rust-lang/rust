const fn failure() {
    panic!("{:?}", 0);
    //~^ ERROR cannot call non-const formatting macro in constant functions
    //~| ERROR cannot call non-const fn `Arguments::<'_>::new_v1` in constant functions
}

const fn print() {
    println!("{:?}", 0);
    //~^ ERROR cannot call non-const formatting macro in constant functions
    //~| ERROR cannot call non-const fn `Arguments::<'_>::new_v1` in constant functions
    //~| ERROR cannot call non-const fn `_print` in constant functions
}

fn main() {}
