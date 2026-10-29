use std::ops::Deref;

struct NotCopy {
    inner: bool
}

impl NotCopy {
    fn inner_method(&self) {}
}

struct Foo {
    first: NotCopy,
    second: NotCopy
}

impl Deref for Foo {
    type Target = NotCopy;
    fn deref(&self) -> &NotCopy {
        &self.second
    }
}

fn use_field(val: Foo) {
    let _val = val.first;
    val.inner; //~ ERROR borrow of
}

fn use_method(val: Foo) {
    let _val = val.first;
    val.inner_method(); //~ ERROR borrow of
}

fn main() {}
