// @has manual/struct.Foo.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//code' 'impl<T> Sync for \
// Foo<T> where T: Sync'
//
// @has - '//*[@id="implementations-list"]/*[@class="impl"]//code' \
// 'impl<T> Send for Foo<T>'
//
// @count - '//*[@id="implementations-list"]/*[@class="impl"]' 1
// @count - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]' 4
pub struct Foo<T> {
    field: T,
}

unsafe impl<T> Send for Foo<T> {}
