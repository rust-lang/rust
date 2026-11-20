pub struct Inner<T: Copy> {
    field: *mut T,
}

//@ has negative/struct.Outer.html
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<T> !Send for Outer<T>"
//
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<T> !Sync for Outer<T>"
pub struct Outer<T: Copy> {
    inner_field: Inner<T>,
}
