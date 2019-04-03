use std::marker::PhantomData;

// @has useless_lifetime_bound/struct.Scope.html
// @!has - '//*[@class="rust struct"]' "'env: 'env"
pub struct Scope<'env> {
    _marker: PhantomData<&'env mut &'env ()>,
}
