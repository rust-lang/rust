#![feature(rustc_attrs)]

// This test checks that a warning occurs with migrate mode.

#[rustc_error]
fn main() {
    //~^ ERROR compilation successful
    let mut x = 0;
    || {
        || {
        //~^ WARNING captured variable cannot escape `FnMut` closure body
        //~| WARNING this error has been downgraded to a warning
        //~| WARNING this warning will become a hard error in the future
            let _y = &mut x;
        }
    };
}
