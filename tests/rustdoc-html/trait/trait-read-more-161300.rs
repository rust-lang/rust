// regression test for <https://github.com/rust-lang/rust/issues/161300>
// ensures the "read more" link exists in various circumstances.
#![crate_name = "foo"]

//@ has 'foo/struct.MyStruct.html'
pub trait MyTrait {
    /// First
    ///
    /// Next paragraph
    //@ has - '//a[@href="trait.MyTrait.html#method.first"]' 'Read more'
    fn first() {}

    /// Second
    ///
    /// ---
    ///
    /// This method is experimental!
    ///
    /// ---
    ///
    /// Next paragraph
    //@ has - '//a[@href="trait.MyTrait.html#method.second"]' 'Read more'
    fn second() {}
}

pub struct MyStruct;
impl MyTrait for MyStruct {}
