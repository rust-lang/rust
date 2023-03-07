// @has manual/struct.Foo.html
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// 'impl<T> Sync for Foo<T>where T: Sync'
//
// @has - '//*[@id="trait-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// 'impl<T> Send for Foo<T>'
//
// @count - '//*[@id="trait-implementations-list"]//*[@class="impl"]' 1
// @count - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]' 4
pub struct Foo<T> {
    field: T,
}

unsafe impl<T> Send for Foo<T> {}
