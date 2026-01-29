//@ has basic/struct.Foo.html
//@ has - '//h3[@class="code-header"]' 'impl<T> Send for Foo<T>where T: Send'
//@ has - '//h3[@class="code-header"]' 'impl<T> Sync for Foo<T>where T: Sync'
//@ count - '//*[@id="implementations-list"]//*[@class="impl"]' 0
//@ count - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]' 7
// The number here will need updating when new auto traits are added:     ^
pub struct Foo<T> {
    field: T,
}
