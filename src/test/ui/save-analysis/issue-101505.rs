// compile-flags: -Zsave-analysis

// Check that this doesn't loop infinitely.

fn a(self) {} //~ ERROR `self` parameter is only allowed in associated functions

fn main() {}
