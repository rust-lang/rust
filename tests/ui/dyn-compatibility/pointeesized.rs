//@ run-pass
//! This test and `sized-*.rs` and `metasized.rs` test that dyn-compatibility correctly
//! handles sizedness traits, which are special in several parts of the compiler.
#![feature(sized_hierarchy)]
// PointeeSized is effectively removed before reaching the trait solver,
// so it's as though it wasn't even mentioned in the trait list.
use std::marker::PointeeSized;

fn main() {
    let dyn_ref: &(dyn PointeeSized + Send) = &42;
    let dyn_ref: &dyn Send = dyn_ref;
    let _dyn_ref: &(dyn PointeeSized + Send) = dyn_ref;
    assert_eq!(
        std::any::TypeId::of::<dyn Send>(),
        std::any::TypeId::of::<dyn PointeeSized + Send>(),
    );
}
