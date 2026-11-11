use std::borrow::Borrow;
pub const F: for<'a> fn(&'a &'static String) -> &'a str = <&'static String as Borrow<str>>::borrow;
//~^ ERROR E0277
fn main() {}
