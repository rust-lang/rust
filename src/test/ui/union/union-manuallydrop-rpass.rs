#![allow(dead_code)]
// run-pass

use std::mem::needs_drop;
use std::mem::ManuallyDrop;

struct NeedDrop;

impl Drop for NeedDrop {
    fn drop(&mut self) {}
}

union UnionOk1<T> {
    empty: (),
    value: ManuallyDrop<T>,
}

union UnionOk2 {
    value: ManuallyDrop<NeedDrop>,
}

#[allow(dead_code)]
union UnionOk3<T: Copy> {
    empty: (),
    value: T,
}

trait Foo { }

trait ImpliesCopy : Copy { }

#[allow(dead_code)]
union UnionOk4<T: ImpliesCopy> {
    value: T,
}

fn main() {
    // NeedDrop should not make needs_drop true
    assert!(!needs_drop::<UnionOk1<NeedDrop>>());
    assert!(!needs_drop::<UnionOk3<&dyn Foo>>());
}
