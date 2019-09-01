// compile-flags: -C overflow-checks=on -O

#![deny(const_err)]

fn main() {
    println!("{}", 0u32 - 1);
    //~^ ERROR attempt to subtract with overflow
    let _x = 0u32 - 1;
    //~^ ERROR attempt to subtract with overflow
    println!("{}", 1/(1-1));
    //~^ ERROR attempt to divide by zero [const_err]
    //~| ERROR reaching this expression at runtime will panic or abort [const_err]
    let _x = 1/(1-1);
    //~^ ERROR const_err
    //~| ERROR const_err
    println!("{}", 1/(false as u32));
    //~^ ERROR attempt to divide by zero [const_err]
    //~| ERROR reaching this expression at runtime will panic or abort [const_err]
    let _x = 1/(false as u32);
    //~^ ERROR const_err
    //~| ERROR const_err
}
