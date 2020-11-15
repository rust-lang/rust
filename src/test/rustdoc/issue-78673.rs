#![crate_name = "issue_78673"]

pub trait Something {}

pub trait AnAmazingTrait {}

impl<T: Something> AnAmazingTrait for T {}

// @has 'issue_78673/struct.MyStruct.html'
// @has  - '//*[@class="impl"]' 'AnAmazingTrait for MyStruct'
// @!has - '//*[@class="impl"]' 'AnAmazingTrait for T'
pub struct MyStruct;

impl AnAmazingTrait for MyStruct {}

// generic structs may have _both_ specific and blanket impls that apply

// @has 'issue_78673/struct.AnotherStruct.html'
// @has - '//*[@class="impl"]' 'AnAmazingTrait for AnotherStruct<()>'
// @has - '//*[@class="impl"]' 'AnAmazingTrait for T'
pub struct AnotherStruct<T>(T);

impl<T: Something> Something for AnotherStruct<T> {}
impl AnAmazingTrait for AnotherStruct<()> {}
