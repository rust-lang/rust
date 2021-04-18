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

// @has issue_53812/trait.MyIterator.html '//*[@id="implementors-list"]/details[1]/summary/h3' 'MyStruct<[T; 0]>'
// @has - '//*[@id="implementors-list"]/details[2]/summary/h3' 'MyStruct<[T; 1]>'
// @has - '//*[@id="implementors-list"]/details[3]/summary/h3' 'MyStruct<[T; 2]>'
// @has - '//*[@id="implementors-list"]/details[4]/summary/h3' 'MyStruct<[T; 3]>'
// @has - '//*[@id="implementors-list"]/details[5]/summary/h3' 'MyStruct<[T; 10]>'
array_impls! { 10 3 2 1 0 }
