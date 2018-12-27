pub struct Inner<T: Copy> {
    field: *mut T,
}

// @has negative/struct.Outer.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//code' "impl<T> !Send for \
// Outer<T>"
//
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//code' "impl<T> \
// !Sync for Outer<T>"
pub struct Outer<T: Copy> {
    inner_field: Inner<T>,
}
