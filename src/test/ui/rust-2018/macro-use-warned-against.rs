// aux-build:macro-use-warned-against.rs
// aux-build:macro-use-warned-against2.rs
// build-pass (FIXME(62277): could be check-pass?)

#![warn(macro_use_extern_crate, unused)]

#[macro_use] //~ WARN should be replaced at use sites with a `use` statement
extern crate macro_use_warned_against;
#[macro_use] //~ WARN unused `#[macro_use]`
extern crate macro_use_warned_against2;

fn main() {
    foo!();
}
