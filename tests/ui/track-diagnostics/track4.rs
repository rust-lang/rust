//@ compile-flags: -Z track-diagnostics
//@ dont-require-annotations: NOTE

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"

pub onion {
    //~^ ERROR missing `enum` for enum definition
    //~| NOTE created at
    Owo(u8),
    Uwu(i8),
}

fn main() {}
