// @has basic/struct.Foo.html
// @has - '//code' 'impl<T> Send for Foo<T> where T: Send'
// @has - '//code' 'impl<T> Sync for Foo<T> where T: Sync'
// @count - '//*[@id="implementations-list"]/*[@class="impl"]' 0
// @count - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]' 5
pub struct Foo<T> {
    field: T,
}
