// revisions: normal expanded
//[expanded] check-pass
//[expanded]compile-flags: -Zunpretty=expanded

extern "路濫狼á́́" fn foo() {} //[normal]~ ERROR invalid ABI

fn main() { }
