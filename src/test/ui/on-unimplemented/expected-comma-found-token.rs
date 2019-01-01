// Tests that two closures cannot simultaneously have mutable
// access to the variable, whether that mutable access be used
// for direct assignment or for taking mutable ref. Issue #6801.

#![feature(on_unimplemented)]

#[rustc_on_unimplemented(
    message="the message"
    label="the label" //~ ERROR expected one of `)` or `,`, found `label`
)]
trait T {}

fn main() {  }
