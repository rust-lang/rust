#![crate_name = "foo"]

use std::ops::Add;
use std::fmt::Display;

//@count foo/fn.function_with_a_really_long_name.html //pre/br 2
pub fn function_with_a_really_long_name(parameter_one: i32,
                                        parameter_two: i32)
                                        -> Option<i32> {
    Some(parameter_one + parameter_two)
}

//@count foo/fn.short_name.html //pre/br 0
pub fn short_name(param: i32) -> i32 { param + 1 }

//@count foo/fn.where_clause.html //pre/br 4
pub fn where_clause<T, U>(param_one: T,
                          param_two: U)
    where T: Add<U> + Display + Copy,
          U: Add<T> + Display + Copy,
          T::Output: Display + Add<U::Output> + Copy,
          <T::Output as Add<U::Output>>::Output: Display,
          U::Output: Display + Copy
{
    let x = param_one + param_two;
    println!("{} + {} = {}", param_one, param_two, x);
    let y = param_two + param_one;
    println!("{} + {} = {}", param_two, param_one, y);
    println!("{} + {} = {}", x, y, x + y);
}
