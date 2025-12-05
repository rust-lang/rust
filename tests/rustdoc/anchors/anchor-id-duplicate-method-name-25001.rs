// https://github.com/rust-lang/rust/issues/25001
#![crate_name="issue_25001"]

//@ has issue_25001/struct.Foo.html
pub struct Foo<T>(T);

pub trait Bar {
    type Item;

    fn quux(self);
}

impl Foo<u8> {
    //@ has - '//*[@id="method.pass"]//h4[@class="code-header"]' 'fn pass()'
    pub fn pass() {}
}
impl Foo<u16> {
    //@ has - '//*[@id="method.pass-1"]//h4[@class="code-header"]' 'fn pass() -> usize'
    pub fn pass() -> usize { 42 }
}
impl Foo<u32> {
    //@ has - '//*[@id="method.pass-2"]//h4[@class="code-header"]' 'fn pass() -> isize'
    pub fn pass() -> isize { 42 }
}

impl<T> Bar for Foo<T> {
    //@ has - '//*[@id="associatedtype.Item"]//h4[@class="code-header"]' 'type Item = T'
    type Item=T;

    //@ has - '//*[@id="method.quux"]//h4[@class="code-header"]' 'fn quux(self)'
    fn quux(self) {}
}
impl<'a, T> Bar for &'a Foo<T> {
    //@ has - '//*[@id="associatedtype.Item-1"]//h4[@class="code-header"]' "type Item = &'a T"
    type Item=&'a T;

    //@ has - '//*[@id="method.quux-1"]//h4[@class="code-header"]' 'fn quux(self)'
    fn quux(self) {}
}
impl<'a, T> Bar for &'a mut Foo<T> {
    //@ has - '//*[@id="associatedtype.Item-2"]//h4[@class="code-header"]' "type Item = &'a mut T"
    type Item=&'a mut T;

    //@ has - '//*[@id="method.quux-2"]//h4[@class="code-header"]' 'fn quux(self)'
    fn quux(self) {}
}
