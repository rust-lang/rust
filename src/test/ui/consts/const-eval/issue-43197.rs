// build-fail

#![warn(const_err)]

const fn foo(x: u32) -> u32 {
    x
}

fn main() {
    const X: u32 = 0 - 1;
    //~^ WARN any use of this value will cause
    //~| WARN this was previously accepted by the compiler but is being phased out
    const Y: u32 = foo(0 - 1);
    //~^ WARN any use of this value will cause
    //~| WARN this was previously accepted by the compiler but is being phased out
    println!("{} {}", X, Y);
    //~^ ERROR evaluation of constant value failed
    //~| ERROR evaluation of constant value failed
    //~| WARN erroneous constant used [const_err]
    //~| WARN erroneous constant used [const_err]
    //~| WARN this was previously accepted by the compiler but is being phased out
    //~| WARN this was previously accepted by the compiler but is being phased out
}
