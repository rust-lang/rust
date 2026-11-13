//@ revisions: normal expanded
//@[expanded] check-pass
//@[expanded]compile-flags: -Zunpretty=expanded
//@ edition: 2015

extern "路濫狼á́́" fn foo() {} //[normal]~ ERROR invalid ABI

fn main() { }
