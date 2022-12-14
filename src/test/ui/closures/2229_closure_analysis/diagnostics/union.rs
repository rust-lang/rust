// edition:2021

// Test that we point to the correct location that results a union being captured.
// Union is special because it can't be disjointly captured.

union A {
    y: u32,
    x: (),
}

fn main() {
    let mut a = A { y: 1 };
    let mut c = || {
    //~^ borrow of `a.y` occurs here
        let _ = unsafe { &a.y };
        let _ = &mut a;
        //~^ borrow occurs due to use in closure
        let _ = unsafe { &mut a.y };
    };
    a.y = 1;
    //~^ cannot assign to `a.y` because it is borrowed [E0506]
    //~| assignment to borrowed `a.y` occurs here
    c();
    //~^ borrow later used here
}
