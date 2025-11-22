#![feature(type_info)]

//@ run-pass

use std::mem::type_info::{Type, TypeKind};

fn assert_tuple_arity<T: 'static, const N: usize>() {
    const {
        match &Type::of::<T>().kind {
            TypeKind::Tuple(tup) => {
                assert!(tup.fields.len() == N);
            }
            _ => unreachable!(),
        }
    }
}

fn main() {
    assert_tuple_arity::<(), 0>();
    assert_tuple_arity::<(u8,), 1>();
    assert_tuple_arity::<(u8, u8), 2>();
    const {
        match &Type::of::<(u8, u8)>().kind {
            TypeKind::Tuple(tup) => {
                let [a, b] = tup.fields else { unreachable!() };
                assert!(a.offset == 0);
                assert!(b.offset == 1);
                match (&a.ty.info().kind, &b.ty.info().kind) {
                    (TypeKind::Leaf, TypeKind::Leaf) => {}
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }
}
