// Regression test for #87456.

fn take_mut(_val: impl FnMut()) {}

fn main() {
    let val = String::new();
    //~^ NOTE: captured outer variable
    take_mut(|| {
    //~^ NOTE: captured by this `FnMut` closure
        let _foo: String = val;
        //~^ ERROR: cannot move out of `val`, a captured variable in an `FnMut` closure [E0507]
        //~| NOTE: move occurs because
    })
}
