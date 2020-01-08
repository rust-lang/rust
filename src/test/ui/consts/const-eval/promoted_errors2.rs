// build-pass
// compile-flags: -C overflow-checks=on -O

#![warn(const_err)]

fn main() {
    println!("{}", 0u32 - 1);
    //~^ WARN attempt to subtract with overflow
    let _x = 0u32 - 1;
    //~^ WARN attempt to subtract with overflow
    println!("{}", 1 / (1 - 1));
    //~^ WARN attempt to divide by zero [const_err]
    //~| WARN const_err
    //~| WARN erroneous constant used [const_err]
    let _x = 1 / (1 - 1);
    //~^ WARN const_err
    println!("{}", 1 / (false as u32));
    //~^ WARN attempt to divide by zero [const_err]
    //~| WARN const_err
    //~| WARN erroneous constant used [const_err]
    let _x = 1 / (false as u32);
    //~^ WARN const_err
}
