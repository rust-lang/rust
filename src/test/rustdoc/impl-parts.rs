#![feature(optin_builtin_traits)]

pub auto trait AnOibit {}

pub struct Foo<T> { field: T }

// @has impl_parts/struct.Foo.html '//*[@class="impl"]//code' \
//     "impl<T: Clone> !AnOibit for Foo<T> where T: Sync,"
// @has impl_parts/trait.AnOibit.html '//*[@class="item-list"]//code' \
//     "impl<T: Clone> !AnOibit for Foo<T> where T: Sync,"
impl<T: Clone> !AnOibit for Foo<T> where T: Sync {}
