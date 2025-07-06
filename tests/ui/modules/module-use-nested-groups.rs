//! Checks complex `use` syntax and availability of types across nested modules.

//@ run-pass

mod a {
    pub enum B {}

    pub mod d {
        pub enum E {}
        pub enum F {}

        pub mod g {
            pub enum H {}
            pub enum I {}
        }
    }
}

// Test every possible part of the syntax
// Test a more common use case
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use a::B;
use a::d::g::H;
use a::d::{self, *};

fn main() {
    let _: B;
    let _: E;
    let _: F;
    let _: H;
    let _: d::g::I;

    let _: Arc<AtomicBool>;
    let _: Ordering;
}
