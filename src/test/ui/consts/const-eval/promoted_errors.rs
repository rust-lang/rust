#![warn(const_err)]

// compile-pass
// compile-flags: -O
fn main() {
    println!("{}", 0u32 - 1);
    let _x = 0u32 - 1;
    //~^ WARN const_err
    println!("{}", 1/(1-1));
    //~^ WARN const_err
    let _x = 1/(1-1);
    //~^ WARN const_err
    //~| WARN const_err
    println!("{}", 1/(false as u32));
    //~^ WARN const_err
    let _x = 1/(false as u32);
    //~^ WARN const_err
    //~| WARN const_err
}
