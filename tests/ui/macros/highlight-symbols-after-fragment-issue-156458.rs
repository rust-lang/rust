// Make sure accepted characters in the macro syntax warning message is highlighted in magenta.

//@ compile-flags: --error-format=human --color=always
//@ edition:2018

macro_rules! m {
    ($e:expr ~ $id:ident) => {};
}

fn main() {}

//~? RAW [1m[35m`=>`[0m, [1m[35m`,`[0m or [1m[35m`;`[0m
