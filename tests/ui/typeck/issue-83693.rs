// Regression test for the ICE described in #83693.

#![feature(fn_traits)]
#![crate_type="lib"]

impl F {
//~^ ERROR: cannot find type `F` in this scope [E0412]
    fn call() {
        <Self as Fn(&TestResult)>::call
        //~^ ERROR: cannot find type `TestResult` in this scope [E0412]
        //~| ERROR associated item constraints are not allowed here [E0229]
    }
}

fn call() {
    <x as Fn(&usize)>::call
    //~^ ERROR: cannot find type `x` in this scope [E0412]
    //~| ERROR: associated item constraints are not allowed here [E0229]
}
