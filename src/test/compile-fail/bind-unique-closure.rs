// -*- rust -*-

// I originally made this test to ensure that bind does the right
// thing when binding a unique closure (which is to copy the closure,
// I suppose?).  But I've since decided it's not worth the effort, and
// so I just made it a simple error.  But I left the test as is in
// case we ever decide that bind should work with unique closures,
// though a simpler test would suffice for now.

fn make_addr(-x: ~int) -> fn~() -> uint {
    (fn~[move x]() -> uint { ptr::addr_of(*x) as uint })
}

fn main() {
    let x = ~3;
    let a = ptr::addr_of(*x) as uint;
    let adder: fn~() -> uint = make_addr(x);
    let bound_adder: fn~() -> uint = bind adder();
    //!^ ERROR cannot bind fn~ closures
    assert adder() == a;
    assert bound_adder() != a;
}
