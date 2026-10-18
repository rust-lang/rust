use std::marker::PhantomData;

//@ has useless_lifetime_bound/struct.Scope.html
//@ has - '//*[@class="rust item-decl"]' "pub struct Scope<'env>"
pub struct Scope<'env> {
    _marker: PhantomData<&'env mut &'env ()>,
}

//@ has useless_lifetime_bound/struct.SomeStruct.html
//@ has - '//*[@class="rust item-decl"]' "pub struct SomeStruct<'a, T: 'a>"
pub struct SomeStruct<'a, T: 'a> {
    _marker: PhantomData<&'a T>,
}
