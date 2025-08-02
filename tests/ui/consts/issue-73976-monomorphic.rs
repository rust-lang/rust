//@ ignore-backends: gcc
//@ check-pass
//
// This test is complement to the test in issue-73976-polymorphic.rs.
// In that test we ensure that polymorphic use of type_id and type_name in patterns
// will be properly rejected. This test will ensure that monomorphic use of these
// would not be wrongly rejected in patterns.

#![feature(const_type_name)]
#![feature(const_trait_impl)]
#![feature(const_cmp)]

use std::any::{self, TypeId};

pub struct GetTypeId<T>(T);

impl<T: 'static> GetTypeId<T> {
    pub const VALUE: TypeId = TypeId::of::<T>();
}

const fn check_type_id<T: 'static>() -> bool {
    GetTypeId::<T>::VALUE == GetTypeId::<usize>::VALUE
}

pub struct GetTypeNameLen<T>(T);

impl<T: 'static> GetTypeNameLen<T> {
    pub const VALUE: usize = any::type_name::<T>().len();
}

const fn check_type_name_len<T: 'static>() -> bool {
    matches!(GetTypeNameLen::<T>::VALUE, GetTypeNameLen::<usize>::VALUE)
}

fn main() {
    assert!(check_type_id::<usize>());
    assert!(check_type_name_len::<usize>());
}
