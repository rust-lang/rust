//@ aux-build:tdticc_coherence_lib.rs

// Test that we do not consider associated types to be sendable without
// some applicable trait bound (and we don't ICE).

#![feature(negative_impls)]

extern crate tdticc_coherence_lib as lib;

use lib::DefaultedTrait;

struct A;
impl DefaultedTrait for (A,) {} //~ ERROR E0117

struct B;
impl !DefaultedTrait for (B,) {} //~ ERROR E0117

struct C;
struct D<T>(T);
impl DefaultedTrait for Box<C> {} //~ ERROR E0321
impl DefaultedTrait for lib::Something<C> {} //~ ERROR E0117
impl DefaultedTrait for D<C> {} // OK

fn main() {}
