pub trait MyIterator {
}

pub struct MyStruct<T>(T);

macro_rules! array_impls {
    ($($N:expr)+) => {
        $(
            impl<'a, T> MyIterator for &'a MyStruct<[T; $N]> {
            }
        )+
    }
}

// @has issue_53812/trait.MyIterator.html '//*[@id="implementors-list"]//h3[1]' 'MyStruct<[T; 0]>'
// @has - '//*[@id="implementors-list"]//h3[2]' 'MyStruct<[T; 1]>'
// @has - '//*[@id="implementors-list"]//h3[3]' 'MyStruct<[T; 2]>'
// @has - '//*[@id="implementors-list"]//h3[4]' 'MyStruct<[T; 3]>'
// @has - '//*[@id="implementors-list"]//h3[5]' 'MyStruct<[T; 10]>'
array_impls! { 10 3 2 1 0 }
