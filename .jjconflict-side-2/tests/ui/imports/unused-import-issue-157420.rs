//@ check-pass
//@ edition: 2018..

#![crate_type = "lib"] // needed to enable doc link collection
#![warn(unused_imports)]

pub use inner::*;
use crate::outer::*;

mod outer {
    pub mod inner {
        pub trait Trait {} // must be a trait
    }

    pub use inner::*;
}

/// [A::assoc] // needed to force collection of traits in scope, without filter on assoc item name
pub struct A;
