#![feature(optin_builtin_traits)]

// @matches negative_impl/struct.Alpha.html '//pre' "pub struct Alpha"
pub struct Alpha;
// @matches negative_impl/struct.Bravo.html '//pre' "pub struct Bravo<B>"
pub struct Bravo<B>(B);

// @matches negative_impl/struct.Alpha.html '//*[@class="impl"]//code' "impl !Send for Alpha"
impl !Send for Alpha {}

// @matches negative_impl/struct.Bravo.html '//*[@class="impl"]//code' "impl<B> !Send for Bravo<B>"
impl<B> !Send for Bravo<B> {}
