//@ edition: 2018
//@ aux-build: issue-114682-5-extern-1.rs
//@ compile-flags: --extern issue_114682_5_extern_1

pub mod p {
    pub use crate::types::*;
    pub use crate::*;
}
mod types {
    pub mod issue_114682_5_extern_1 {}
}

pub use issue_114682_5_extern_1;
