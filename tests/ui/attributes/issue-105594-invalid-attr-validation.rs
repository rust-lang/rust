// This checks that the attribute validation ICE in issue #105594 doesn't
// recur.
//
//@ ignore-thumbv8m.base-none-eabi
#![feature(cmse_nonsecure_entry)]

fn main() {}

#[track_caller] //~ ERROR attribute should be applied to a function
static _A: () = ();

#[cmse_nonsecure_entry] //~ ERROR attribute should be applied to a function
static _B: () = (); //~| ERROR #[cmse_nonsecure_entry]` is only valid for targets
