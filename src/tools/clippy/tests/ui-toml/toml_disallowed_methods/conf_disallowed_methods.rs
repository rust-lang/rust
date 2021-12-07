#![warn(clippy::disallowed_methods)]

extern crate regex;
use regex::Regex;

fn main() {
    let re = Regex::new(r"ab.*c").unwrap();
    re.is_match("abc");

    let a = vec![1, 2, 3, 4];
    a.iter().sum::<i32>();
}
