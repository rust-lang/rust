#![crate_name = "foo"]

// @has foo/struct.SomeStruct.html '//*[@class="sidebar-links"]/a[@href="#method.some_fn-1"]' \
//          "some_fn"
pub struct SomeStruct<T> { _inner: T }

impl SomeStruct<()> {
    pub fn some_fn(&self) {}
}

impl SomeStruct<usize> {
    pub fn some_fn(&self) {}
}
