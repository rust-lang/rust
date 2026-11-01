// rust-lang/rust#45696: This test checks the compiler won't infinite loop when
// you declare a variable of type `struct A(Box<A>, ...);` (which is impossible
// to construct but *is* possible to declare; see also issues #4287, #44933,
// and #52852).

//@ run-pass

// This test has structs and functions that are by definition unusable
// all over the place, so just go ahead and allow dead_code
#![allow(dead_code)]

// direct regular recursion with indirect ownership via box
struct C { field: Box<C> }

// direct non-regular recursion with indirect ownership via box
struct D { field: Box<(D, D)> }

// indirect regular recursion with indirect ownership via box.
struct E { field: F }
struct F { field: Box<E> }

// indirect non-regular recursion with indirect ownership via box.
struct G { field: (H, H) }
struct H { field: Box<G> }

// These enums are cases that are not currently hit by the
// `visit_terminator_drop` recursion down a type's structural
// definition.
//
// But it seems prudent to include them in this test as variants on
// the above, in that they are similarly non-constructable data types
// with destructors that would diverge.
enum I { One(Box<I>) }
enum J { One(Box<J>), Two(Box<J>) }

fn impossible_to_call_c(_c: C) { }
fn impossible_to_call_d(_d: D) { }
fn impossible_to_call_e(_e: E) { }
fn impossible_to_call_f(_f: F) { }
fn impossible_to_call_g(_g: G) { }
fn impossible_to_call_h(_h: H) { }
fn impossible_to_call_i(_i: I) { }
fn impossible_to_call_j(_j: J) { }

fn main() {

}
