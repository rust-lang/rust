


// use of tail calls causes arg slot leaks, issue #160.
fn inner(str dummy, bool b) { if (b) { be inner(dummy, false); } }

fn main() { inner("hi", true); }