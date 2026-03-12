// gate-test-const_closures

fn main() {
    const { (const || {})() };
    //~^ ERROR: const closures are experimental
    //~| ERROR: the trait bound `{closure@$DIR/gate.rs:4:14: 4:22}: [const] Fn()` is not satisfied
}

macro_rules! e {
    ($e:expr) => {};
}

e!((const || {}));
//~^ ERROR const closures are experimental
