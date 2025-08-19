//@ compile-flags: -Z track-diagnostics
//@ dont-require-annotations: NOTE

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"

fn main() {
    let _moved @ _from = String::from("foo");
    //~^ ERROR use of moved value
    //~| NOTE created at
}
