pub struct Inner<T> {
    field: T,
}

unsafe impl<T> Send for Inner<T>
where
    T: Copy,
{
}

// @has nested/struct.Foo.html
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl has-srclink"]//h3[@class="code-header in-band"]' \
// 'impl<T> Send for Foo<T>where T: Copy'
//
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl has-srclink"]//h3[@class="code-header in-band"]' \
// 'impl<T> Sync for Foo<T>where T: Sync'
pub struct Foo<T> {
    inner_field: Inner<T>,
}
