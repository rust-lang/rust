pub struct Inner<T> {
    field: T,
}

unsafe impl<T> Send for Inner<T>
where
    T: Copy + Send,
{
}

//@ has no_redundancy/struct.Outer.html
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<T> Send for Outer<T>where T: Copy + Send"
pub struct Outer<T> {
    inner_field: Inner<T>,
}
