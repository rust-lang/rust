#![crate_name = "foo"]

use std::fmt::Display;
use std::ops::Add;

//@ matches foo/fn.function_with_a_really_long_name.html '//*[@class="rust item-decl"]//code' "\
//     function_with_a_really_long_name\(\n\
//    \    parameter_one: i32,\n\
//    \    parameter_two: i32,\n\
//    \) -> Option<i32>$"
pub fn function_with_a_really_long_name(parameter_one: i32, parameter_two: i32) -> Option<i32> {
    Some(parameter_one + parameter_two)
}

//@ matches foo/fn.short_name.html '//*[@class="rust item-decl"]//code' \
//     "short_name\(param: i32\) -> i32$"
pub fn short_name(param: i32) -> i32 {
    param + 1
}

//@ matches foo/fn.where_clause.html '//*[@class="rust item-decl"]//code' "\
//     where_clause<T, U>\(param_one: T, param_two: U\)where\n\
//    \    T: Add<U> \+ Display \+ Copy,\n\
//    \    U: Add<T> \+ Display \+ Copy,\n\
//    \    T::Output: Display \+ Add<U::Output> \+ Copy,\n\
//    \    <T::Output as Add<U::Output>>::Output: Display,\n\
//    \    U::Output: Display \+ Copy,$"
pub fn where_clause<T, U>(param_one: T, param_two: U)
where
    T: Add<U> + Display + Copy,
    U: Add<T> + Display + Copy,
    T::Output: Display + Add<U::Output> + Copy,
    <T::Output as Add<U::Output>>::Output: Display,
    U::Output: Display + Copy,
{
    let x = param_one + param_two;
    println!("{} + {} = {}", param_one, param_two, x);
    let y = param_two + param_one;
    println!("{} + {} = {}", param_two, param_one, y);
    println!("{} + {} = {}", x, y, x + y);
}
