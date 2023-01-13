// run-pass

// edition:2018
// pp-exact

fn main() { let _a = (async { }); }
//~^ WARNING unnecessary parentheses around assigned value
