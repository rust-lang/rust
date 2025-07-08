//@ compile-flags: -Z track-diagnostics
//@ dont-require-annotations: NOTE

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"

}
//~^ ERROR unexpected closing delimiter: `}`
//~| NOTE created at
