// Plain doc attribute
#![doc("a")] //~ ERROR

// doc hidden
#![doc(hidden = "a")] //~ ERROR
#![doc(hidden = 12)] //~ ERROR
#![doc(hidden())] //~ ERROR
#![doc(hidden("a"))] //~ ERROR

// Error when `no_crate_inject` is used more than once.
#![doc(test(no_crate_inject))]
#![doc(test(no_crate_inject))] //~ ERROR
