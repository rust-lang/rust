//@ run-pass

#![feature(core_intrinsics)]

use std::any::type_name;

struct Bar<M>(#[allow(dead_code)] M);

impl<M> Bar<M> {
    fn foo(&self) -> &'static str {
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            type_name::<T>()
        }
        type_name_of(f)
    }
}

fn main() {
    assert_eq!(Bar(()).foo(), "issue_61894::Bar<_>::foo::f");
}
