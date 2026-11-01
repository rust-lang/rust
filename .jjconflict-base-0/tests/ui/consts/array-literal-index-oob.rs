//@ build-pass
//@ no-pass-override (test emits codegen-time warnings and verifies that they are not errors)

#![warn(unconditional_panic)]

fn main() {
    &{ [1, 2, 3][4] };
    //~^ WARN operation will panic
}
