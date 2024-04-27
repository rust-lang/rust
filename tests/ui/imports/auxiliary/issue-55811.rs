mod m {}

// These two imports should not conflict when this crate is loaded from some other crate.
use m::{};
use m::{};
