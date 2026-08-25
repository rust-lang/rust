// Regression test for the ICE described in #84768.

#![feature(fn_traits)]
#![crate_type="lib"]

fn transform_mut<F>(f: F) where F: for<'b> FnOnce(&'b mut u8) {
    <F as FnOnce(&mut u8)>::call_once(f, 1)
    //~^ ERROR: associated item constraints are not allowed here [E0229]
    //~| ERROR: mismatched types [E0308]
}
