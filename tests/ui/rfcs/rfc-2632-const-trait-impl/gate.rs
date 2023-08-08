// gate-test-const_closures

fn main() {
    (const || {})();
    //~^ ERROR: const closures are experimental
}

macro_rules! e {
    ($e:expr) => {}
}

e!((const || {}));
//~^ ERROR const closures are experimental
