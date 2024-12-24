const fn failure() {
    panic!("{:?}", 0);
    //~^ ERROR cannot call non-const formatting macro in constant functions
    //~| ERROR cannot call non-const associated function `Arguments::<'_>::new_v1::<1, 1>` in constant functions
}

const fn print() {
    println!("{:?}", 0);
    //~^ ERROR cannot call non-const formatting macro in constant functions
    //~| ERROR cannot call non-const associated function `Arguments::<'_>::new_v1::<2, 1>` in constant functions
    //~| ERROR cannot call non-const function `_print` in constant functions
}

fn main() {}
