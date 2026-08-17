// Tests that variadic functions using `splat` have their splatted argument replaced
// with an ellipsis in documentation. Prior to this test, these functions would be documented
// as having a tuple-like argument, when in reality they're called with the contents of said tuple.

#![crate_name = "foo"]
#![no_std]
#![expect(incomplete_features)]
#![feature(splat)]

//@ has 'foo/fn.my_variadic_function.html' //pre 'pub fn my_variadic_function(…: (u8, u8)) -> u8'
pub fn my_variadic_function(#[rustc_splat] args: (u8, u8)) -> u8 {
    args.0 + args.1
}

pub fn test() {
    // As can be seen here, my_variadic_function actually takes 2 arguments
    assert_eq!(my_variadic_function(1, 2), 3);
}
