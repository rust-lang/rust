//@ check-pass
//@ edition:2024

#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

struct Unpinned;

#[pin_v2]
struct Pinned;

struct Qualified;
struct CoreQualified;

impl Drop for Unpinned {
    fn drop(&pin mut self) {}
}

impl Drop for Pinned {
    fn drop(&pin mut self) {}
}

impl std::ops::Drop for Qualified {
    fn drop(&pin mut self) {}
}

impl core::ops::Drop for CoreQualified {
    fn drop(&pin mut self) {}
}

fn main() {}
