//@ run-pass
// Verify that PartialEq implementations do not break type inference when
// accepting types with different allocators

use std::rc::Rc;
use std::sync::Arc;


fn main() {
    let boxed: Vec<Box<i32>> = vec![];
    assert_eq!(boxed, vec![]);

    let rc: Vec<Rc<i32>> = vec![];
    assert_eq!(rc, vec![]);

    let arc: Vec<Arc<i32>> = vec![];
    assert_eq!(arc, vec![]);
}
