// gate-test-const_closures

fn main() {
    const { (const || {})() };
    //~^ ERROR: const closures are experimental
    //~| ERROR: cannot call conditionally-const closure in constants
    //~| ERROR:  `Fn` is not yet stable as a const trait
}

macro_rules! e {
    ($e:expr) => {};
}

e!((const || {}));
//~^ ERROR const closures are experimental
