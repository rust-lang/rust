#![warn(default_trait_access)]

use std::default::Default as D2;
use std::string;
use std::default;

fn main() {
    let s1: String = Default::default();

    let s2 = String::default();

    let s3: String = D2::default();

    let s4: String = std::default::Default::default();

    let s5 = string::String::default();

    let s6: String = default::Default::default();

    let s7 = std::string::String::default();

    let s8: String = DefaultFactory::make_t_badly();

    let s9: String = DefaultFactory::make_t_nicely();

    println!("[{}] [{}] [{}] [{}] [{}] [{}] [{}] [{}] [{}]", s1, s2, s3, s4, s5, s6, s7, s8, s9);
}

struct DefaultFactory;

impl DefaultFactory {
    pub fn make_t_badly<T: Default>() -> T {
        Default::default()
    }

    pub fn make_t_nicely<T: Default>() -> T {
        T::default()
    }
}
