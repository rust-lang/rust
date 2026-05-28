pub struct Inner<'a, T: 'a> {
    field: &'a T,
}

unsafe impl<'a, T> Send for Inner<'a, T>
where
    'a: 'static,
    T: for<'b> Fn(&'b bool) -> &'a u8,
{}

//@ has lifetimes/struct.Foo.html
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<'c, K> Send for Foo<'c, K>where 'c: 'static, K: for<'b> Fn(&'b bool) -> &'c u8,"
//
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<'c, K> Sync for Foo<'c, K>where K: Sync"
pub struct Foo<'c, K: 'c> {
    inner_field: Inner<'c, K>,
}
