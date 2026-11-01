// regression test for https://github.com/rust-lang/rust/issues/144965

#![crate_name = "foo"]
#![no_std]

#[doc(hidden)]
pub struct MyStruct;

macro_rules! my_macro {
    () => {
        pub fn my_function() {}

        /// Incorrect: [`my_function()`].
        #[doc(inline)]
        pub use $crate::MyStruct;

        /// Correct: [`my_function`].
        pub struct AnotherStruct;
    };
}


pub mod one {
    //@ has 'foo/one/index.html'
    //@ has - '//dl[@class="item-table"]/dd[1]/a[@href="fn.my_function.html"]/code' 'my_function'
    //@ has - '//dl[@class="item-table"]/dd[2]/a[@href="fn.my_function.html"]/code' 'my_function()'
    my_macro!();
}

pub mod two {
    //@ has 'foo/two/index.html'
    //@ has - '//dl[@class="item-table"]/dd[1]/a[@href="fn.my_function.html"]/code' 'my_function'
    //@ has - '//dl[@class="item-table"]/dd[2]/a[@href="fn.my_function.html"]/code' 'my_function()'
    my_macro!();
}
