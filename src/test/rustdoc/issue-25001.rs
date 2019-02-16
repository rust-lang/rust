// @has issue_25001/struct.Foo.html
pub struct Foo<T>(T);

pub trait Bar {
    type Item;

    fn quux(self);
}

impl Foo<u8> {
    // @has - '//*[@id="method.pass"]//code' 'fn pass()'
    // @has - '//code[@id="pass.v"]' 'fn pass()'
    pub fn pass() {}
}
impl Foo<u16> {
    // @has - '//*[@id="method.pass-1"]//code' 'fn pass() -> usize'
    // @has - '//code[@id="pass.v-1"]' 'fn pass() -> usize'
    pub fn pass() -> usize { 42 }
}
impl Foo<u32> {
    // @has - '//*[@id="method.pass-2"]//code' 'fn pass() -> isize'
    // @has - '//code[@id="pass.v-2"]' 'fn pass() -> isize'
    pub fn pass() -> isize { 42 }
}

impl<T> Bar for Foo<T> {
    // @has - '//*[@id="associatedtype.Item"]//code' 'type Item = T'
    type Item=T;

    // @has - '//*[@id="method.quux"]//code' 'fn quux(self)'
    fn quux(self) {}
}
impl<'a, T> Bar for &'a Foo<T> {
    // @has - '//*[@id="associatedtype.Item-1"]//code' "type Item = &'a T"
    type Item=&'a T;

    // @has - '//*[@id="method.quux-1"]//code' 'fn quux(self)'
    fn quux(self) {}
}
impl<'a, T> Bar for &'a mut Foo<T> {
    // @has - '//*[@id="associatedtype.Item-2"]//code' "type Item = &'a mut T"
    type Item=&'a mut T;

    // @has - '//*[@id="method.quux-2"]//code' 'fn quux(self)'
    fn quux(self) {}
}
