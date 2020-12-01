#![warn(clippy::disallowed_method)]

extern crate regex;
use regex::Regex;

fn main() {
    let a = vec![1, 2, 3, 4];
    let re = Regex::new(r"ab.*c").unwrap();

    re.is_match("abc");

    a.iter().sum::<i32>();
}
