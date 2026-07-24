//! Regression test for <https://github.com/rust-lang/rust/issues/38857>.
//! Trying to access locally pub but inaccessible items caused ICE.

fn main() {
    let a = std::sys::imp::process::process_common::StdioPipes { ..panic!() };
    //~^ ERROR: cannot find `imp` in `sys` [E0433]
    //~| ERROR: module `sys` is private [E0603]
}
