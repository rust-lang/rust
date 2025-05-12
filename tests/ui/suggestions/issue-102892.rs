#![allow(dead_code, unused_variables)]

use std::sync::Arc;

#[derive(Debug)]
struct A;
#[derive(Debug)]
struct B;

fn process_without_annot(arc: &Arc<(A, B)>) {
    let (a, b) = **arc; // suggests putting `&**arc` here; with that, fixed!
    //~^ ERROR: cannot move out of an `Arc`
}

fn process_with_annot(arc: &Arc<(A, B)>) {
    let (a, b): (A, B) = &**arc; // suggests putting `&**arc` here too
    //~^ ERROR mismatched types
}

fn process_with_tuple_annot(mutation: &mut (A, B), arc: &Arc<(A, B)>) {
    let (a, b): ((A, B), A) = (&mut *mutation, &(**arc).0); // suggests putting `&**arc` here too
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}

fn main() {}
