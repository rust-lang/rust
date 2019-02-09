fn main() {
    // N.B., this (almost) type-checks when default binding modes are enabled.
    for (ref i,) in [].iter() {
        i.clone();
        //~^ ERROR type annotations needed
    }
}
