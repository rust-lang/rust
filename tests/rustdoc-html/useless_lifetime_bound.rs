use std::marker::PhantomData;

//@ has useless_lifetime_bound/struct.Scope.html
//@ !has - '//*[@class="rust struct"]' "'env: 'env"
pub struct Scope<'env> {
    _marker: PhantomData<&'env mut &'env ()>,
}

//@ has useless_lifetime_bound/struct.Scope.html
//@ !has - '//*[@class="rust struct"]' "T: 'a + 'a"
pub struct SomeStruct<'a, T: 'a> {
    _marker: PhantomData<&'a T>,
}
