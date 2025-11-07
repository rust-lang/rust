//@ run-pass
#![feature(sized_hierarchy)]
// PointeeSized is effectively removed before reaching the trait solver.
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
