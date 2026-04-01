pub struct Inner<'a, T: 'a> {
    field: &'a T,
}

trait MyTrait {
    type MyItem;
}

trait OtherTrait {}

unsafe impl<'a, T> Send for Inner<'a, T>
where
    'a: 'static,
    T: MyTrait<MyItem = bool>,
{
}
unsafe impl<'a, T> Sync for Inner<'a, T>
where
    'a: 'static,
    T: MyTrait,
    <T as MyTrait>::MyItem: OtherTrait,
{
}

//@ has project/struct.Foo.html
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<'c, K> Send for Foo<'c, K>where 'c: 'static, K: MyTrait<MyItem = bool>,"
//
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<'c, K> Sync for Foo<'c, K>where 'c: 'static, K: MyTrait, \
// <K as MyTrait>::MyItem: OtherTrait,"
pub struct Foo<'c, K: 'c> {
    inner_field: Inner<'c, K>,
}
