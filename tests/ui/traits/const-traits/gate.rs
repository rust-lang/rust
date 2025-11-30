// gate-test-const_closures

fn main() {
    (const || {})();
    //~^ ERROR: const closures are experimental
    //~| ERROR: the trait bound `{closure@$DIR/gate.rs:4:6: 4:14}: [const] Fn()` is not satisfied
}

macro_rules! e {
    ($e:expr) => {}
}

e!((const || {}));
//~^ ERROR const closures are experimental
