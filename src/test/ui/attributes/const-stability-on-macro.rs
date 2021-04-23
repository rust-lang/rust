#[rustc_const_stable(feature = "foo", since = "0")]
//~^ ERROR macros cannot have const stability attributes
macro_rules! foo {
    () => {};
}

#[rustc_const_unstable(feature = "bar", issue="none")]
//~^ ERROR macros cannot have const stability attributes
macro_rules! bar {
    () => {};
}

fn main() {}
