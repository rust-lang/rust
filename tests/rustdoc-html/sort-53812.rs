// https://github.com/rust-lang/rust/issues/53812
#![crate_name="foo"]

pub trait MyIterator {}

pub struct MyStruct<T>(T);

macro_rules! array_impls {
    ($($N:expr)+) => {
        $(
            impl<'a, T> MyIterator for &'a MyStruct<[T; $N]> {
            }
        )+
    }
}

//@ has foo/trait.MyIterator.html
//@ has - '//*[@id="implementors-list"]/*[@class="impl"][1]' 'MyStruct<[T; 0]>'
//@ has - '//*[@id="implementors-list"]/*[@class="impl"][2]' 'MyStruct<[T; 1]>'
//@ has - '//*[@id="implementors-list"]/*[@class="impl"][3]' 'MyStruct<[T; 2]>'
//@ has - '//*[@id="implementors-list"]/*[@class="impl"][4]' 'MyStruct<[T; 3]>'
//@ has - '//*[@id="implementors-list"]/*[@class="impl"][5]' 'MyStruct<[T; 10]>'
array_impls! { 10 3 2 1 0 }
