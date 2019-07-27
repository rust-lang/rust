// run-pass
#![feature(core_intrinsics)]
#![feature(const_type_id)]

use std::any::TypeId;

struct A;

static ID_ISIZE: TypeId = TypeId::of::<isize>();

pub fn main() {
    assert_eq!(ID_ISIZE, TypeId::of::<isize>());

    // sanity test of TypeId
    const T: (TypeId, TypeId, TypeId) = (TypeId::of::<usize>(),
                     TypeId::of::<&'static str>(),
                     TypeId::of::<A>());
    let (d, e, f) = (TypeId::of::<usize>(), TypeId::of::<&'static str>(),
                     TypeId::of::<A>());

    assert!(T.0 != T.1);
    assert!(T.0 != T.2);
    assert!(T.1 != T.2);

    assert_eq!(T.0, d);
    assert_eq!(T.1, e);
    assert_eq!(T.2, f);

    // Check fn pointer against collisions
    const F: (TypeId, TypeId) = (TypeId::of::<fn(fn(A) -> A) -> A>(),
            TypeId::of::<fn(fn() -> A, A) -> A>());

    assert!(F.0 != F.1);
}
