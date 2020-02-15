// revisions: debug opt opt_with_overflow_checks
//[debug]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

// build-pass
// ignore-pass (emit codegen-time warnings and verify that they are indeed warnings and not errors)

#![warn(const_err, overflow, panic)]

fn main() {
    println!("{}", 0u32 - 1);
    //[opt_with_overflow_checks,debug]~^ WARN [overflow]
    let _x = 0u32 - 1;
    //~^ WARN [overflow]
    println!("{}", 1 / (1 - 1));
    //~^ WARN [panic]
    //~| WARN panic or abort [const_err]
    //~| WARN erroneous constant used [const_err]
    let _x = 1 / (1 - 1);
    //~^ WARN [panic]
    println!("{}", 1 / (false as u32));
    //~^ WARN [panic]
    //~| WARN panic or abort [const_err]
    //~| WARN erroneous constant used [const_err]
    let _x = 1 / (false as u32);
    //~^ WARN [panic]
}
