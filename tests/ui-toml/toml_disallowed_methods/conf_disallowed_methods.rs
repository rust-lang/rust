#![warn(clippy::disallowed_methods)]

extern crate futures;
extern crate regex;

use futures::stream::{empty, select_all};
use regex::Regex;

fn main() {
    let re = Regex::new(r"ab.*c").unwrap();
    re.is_match("abc");

    let mut a = vec![1, 2, 3, 4];
    a.iter().sum::<i32>();

    a.sort_unstable();

    let _ = 2.0f32.clamp(3.0f32, 4.0f32);
    let _ = 2.0f64.clamp(3.0f64, 4.0f64);

    let indirect: fn(&str) -> Result<Regex, regex::Error> = Regex::new;
    let re = indirect(".").unwrap();

    let in_call = Box::new(f32::clamp);
    let in_method_call = ["^", "$"].into_iter().map(Regex::new);

    // resolve ambiguity between `futures::stream::select_all` the module and the function
    let same_name_as_module = select_all(vec![empty::<()>()]);
}
