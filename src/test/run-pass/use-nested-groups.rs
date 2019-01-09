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
use a::{B, d::{self, *, g::H}};

// Test a more common use case
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

fn main() {
    let _: B;
    let _: E;
    let _: F;
    let _: H;
    let _: d::g::I;

    let _: Arc<AtomicBool>;
    let _: Ordering;
}
