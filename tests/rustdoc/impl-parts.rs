#![feature(negative_impls)]
#![feature(rustc_attrs)]

#[rustc_auto_trait]
pub trait AnAutoTrait {}

pub struct Foo<T> { field: T }

// @has impl_parts/struct.Foo.html '//*[@class="impl"]//h3[@class="code-header"]' \
//     "impl<T> !AnAutoTrait for Foo<T>where T: Sync + Clone,"
// @has impl_parts/trait.AnAutoTrait.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//     "impl<T> !AnAutoTrait for Foo<T>where T: Sync + Clone,"
impl<T: Clone> !AnAutoTrait for Foo<T> where T: Sync {}
