// @has manual/struct.Foo.html
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl has-srclink"]//h3[@class="code-header in-band"]' \
// 'impl<T> Sync for Foo<T>where T: Sync'
//
// @has - '//*[@id="trait-implementations-list"]//*[@class="impl has-srclink"]//h3[@class="code-header in-band"]' \
// 'impl<T> Send for Foo<T>'
//
// @count - '//*[@id="trait-implementations-list"]//*[@class="impl has-srclink"]' 1
// @count - '//*[@id="synthetic-implementations-list"]//*[@class="impl has-srclink"]' 4
pub struct Foo<T> {
    field: T,
}

unsafe impl<T> Send for Foo<T> {}
