// Regression test for hashing involving canonical variables.  In this
// test -- which has an intensional error -- the type of the value
// being dropped winds up including a type variable. Canonicalization
// would then produce a `?0` which -- in turn -- triggered an ICE in
// hashing.

// revisions:cfail1

fn main() {
    println!("Hello, world! {}",*thread_rng().choose(&[0, 1, 2, 3]).unwrap());
    //[cfail1]~^ ERROR cannot find function `thread_rng`
}
