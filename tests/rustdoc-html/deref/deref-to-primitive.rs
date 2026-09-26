#![crate_name = "foo"]

//@ has 'foo/struct.Foo.html'
//@ has - '//*[@id="deref-methods-str"]' 'Methods from Deref<Target = str>'
//@ has - '//*[@id="deref-methods-str-1"]//*[@id="method.len"]/h4' \
//        'pub '
pub struct Foo(&'static str);

impl std::ops::Deref for Foo {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
