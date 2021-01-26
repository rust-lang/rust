// revisions: noopt opt opt_with_overflow_checks
//[noopt]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

// build-pass
// ignore-pass (test emits codegen-time warnings and verifies that they are not errors)

#![warn(const_err, arithmetic_overflow, unconditional_panic)]

fn main() {
    println!("{}", 0u32 - 1);
    //[opt_with_overflow_checks,noopt]~^ WARN [arithmetic_overflow]
    let _x = 0u32 - 1;
    //~^ WARN [arithmetic_overflow]
    println!("{}", 1 / (1 - 1));
    //~^ WARN [unconditional_panic]
    //~| WARN panic or abort [const_err]
    //~| WARN erroneous constant used [const_err]
    let _x = 1 / (1 - 1);
    //~^ WARN [unconditional_panic]
    println!("{}", 1 / (false as u32));
    //~^ WARN [unconditional_panic]
    //~| WARN panic or abort [const_err]
    //~| WARN erroneous constant used [const_err]
    let _x = 1 / (false as u32);
    //~^ WARN [unconditional_panic]
}
