// https://github.com/rust-lang/rust/issues/32890
#![crate_name="issue_32890"]

//@ has issue_32890/struct.Foo.html
pub struct Foo<T>(T);

impl Foo<u8> {
    //@ has - '//a[@href="#method.pass"]' 'pass'
    pub fn pass() {}
}

impl Foo<u16> {
    //@ has - '//a[@href="#method.pass-1"]' 'pass'
    pub fn pass() {}
}

impl Foo<u32> {
    //@ has - '//a[@href="#method.pass-2"]' 'pass'
    pub fn pass() {}
}
