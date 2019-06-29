// compile-flags: -O

#![deny(const_err)]

fn main() {
    println!("{}", 0u32 - 1);
    let _x = 0u32 - 1;
    //~^ ERROR this expression will panic at runtime [const_err]
    println!("{}", 1/(1-1));
    //~^ ERROR this expression will panic at runtime [const_err]
    //~| ERROR attempt to divide by zero [const_err]
    //~| ERROR reaching this expression at runtime will panic or abort [const_err]
    let _x = 1/(1-1);
    //~^ ERROR const_err
    //~| ERROR const_err
    println!("{}", 1/(false as u32));
    //~^ ERROR this expression will panic at runtime [const_err]
    //~| ERROR attempt to divide by zero [const_err]
    //~| ERROR reaching this expression at runtime will panic or abort [const_err]
    let _x = 1/(false as u32);
    //~^ ERROR const_err
    //~| ERROR const_err
}
