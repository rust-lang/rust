// test for `doc(hidden)` with impl parameters in the same crate.
#![crate_name = "foo"]

#[doc(hidden)]
pub enum HiddenType {}

#[doc(hidden)]
pub trait HiddenTrait {}

pub enum MyLibType {}

//@ !has foo/enum.MyLibType.html '//*[@id="impl-From%3CHiddenType%3E"]' 'impl From<HiddenType> for MyLibType'
impl From<HiddenType> for MyLibType {
    fn from(it: HiddenType) -> MyLibType {
        match it {}
    }
}

pub struct T<T>(T);

//@ !has foo/enum.MyLibType.html '//*[@id="impl-From%3CT%3CT%3CT%3CT%3CHiddenType%3E%3E%3E%3E%3E"]' 'impl From<T<T<T<T<HiddenType>>>>> for MyLibType'
impl From<T<T<T<T<HiddenType>>>>> for MyLibType {
    fn from(it: T<T<T<T<HiddenType>>>>) -> MyLibType {
        todo!()
    }
}

//@ !has foo/enum.MyLibType.html '//*[@id="impl-HiddenTrait"]' 'impl HiddenTrait for MyLibType'
impl HiddenTrait for MyLibType {}

//@ !has foo/struct.T.html '//*[@id="impl-From%3CMyLibType%3E"]' 'impl From<MyLibType> for T<T<T<T<HiddenType>>>>'
impl From<MyLibType> for T<T<T<T<HiddenType>>>> {
    fn from(it: MyLibType) -> T<T<T<T<HiddenType>>>> {
        match it {}
    }
}
