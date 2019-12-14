// build-fail
// compile-flags: -O

#![deny(const_err)]

fn main() {
    println!("{}", 0u32 - 1);
    let _x = 0u32 - 1;
    //~^ ERROR const_err
    println!("{}", 1/(1-1));
    //~^ ERROR attempt to divide by zero [const_err]
    //~| ERROR const_err
    let _x = 1/(1-1);
    //~^ ERROR const_err
    println!("{}", 1/(false as u32));
    //~^ ERROR attempt to divide by zero [const_err]
    //~| ERROR const_err
    let _x = 1/(false as u32);
    //~^ ERROR const_err
}
