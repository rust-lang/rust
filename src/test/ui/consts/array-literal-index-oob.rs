// build-pass
// ignore-pass (test emits codegen-time warnings and verifies that they are not errors)

#![warn(const_err, unconditional_panic)]

fn main() {
    &{ [1, 2, 3][4] };
    //~^ WARN operation will panic
    //~| WARN reaching this expression at runtime will panic or abort
    //~| WARN erroneous constant used [const_err]
}
