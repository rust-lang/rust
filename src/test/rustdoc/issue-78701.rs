#![crate_name = "foo"]

// This test ensures that if a blanket impl has the same ID as another impl, it'll
// link to the blanket impl and not the other impl. Basically, we're checking if
// the ID is correctly derived.

// @has 'foo/struct.AnotherStruct.html'
// @count - '//*[@class="sidebar"]//a[@href="#impl-AnAmazingTrait"]' 1
// @count - '//*[@class="sidebar"]//a[@href="#impl-AnAmazingTrait-1"]' 1

pub trait Something {}

pub trait AnAmazingTrait {}

impl<T: Something> AnAmazingTrait for T {}

pub struct AnotherStruct<T>(T);

impl<T: Something> Something for AnotherStruct<T> {}
impl AnAmazingTrait for AnotherStruct<()> {}
