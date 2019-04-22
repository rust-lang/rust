// There isn't a great way to test feature(nll), since it just disables migrate
// mode and changes some error messages. We just test for migrate mode.

// Don't use compare-mode=nll, since that turns on NLL.
// ignore-compare-mode-nll

#![feature(rustc_attrs)]

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let mut x = (33, &0);

    let m = &mut x;
    let p = &*x.1;
    //~^ WARNING cannot borrow
    //~| WARNING this error has been downgraded to a warning
    //~| WARNING this warning will become a hard error in the future
    m;
}
