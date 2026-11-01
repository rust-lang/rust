//@ edition:2021

// Test that we point to the correct location that results a union being captured.
// Union is special because it can't be disjointly captured.

union A {
    y: u32,
    x: (),
}

fn main() {
    let mut a = A { y: 1 };
    let mut c = || {
    //~^ NOTE `a.y` is borrowed here
        let _ = unsafe { &a.y };
        let _ = &mut a;
        //~^ NOTE borrow occurs due to use in closure
        let _ = unsafe { &mut a.y };
    };
    a.y = 1;
    //~^ ERROR cannot assign to `a.y` because it is borrowed [E0506]
    //~| NOTE `a.y` is assigned to here
    c();
    //~^ NOTE borrow later used here
}
