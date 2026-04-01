#![allow(unused)]
#![warn(clippy::manual_main_separator_str)]

use std::path::MAIN_SEPARATOR;

fn len(s: &str) -> usize {
    s.len()
}

struct U<'a> {
    f: &'a str,
    g: &'a String,
}

struct V<T> {
    f: T,
}

fn main() {
    // Should lint
    let _: &str = &MAIN_SEPARATOR.to_string();
    //~^ manual_main_separator_str
    let _ = len(&MAIN_SEPARATOR.to_string());
    //~^ manual_main_separator_str
    let _: Vec<u16> = MAIN_SEPARATOR.to_string().encode_utf16().collect();
    //~^ manual_main_separator_str

    // Should lint for field `f` only
    let _ = U {
        f: &MAIN_SEPARATOR.to_string(),
        //~^ manual_main_separator_str
        g: &MAIN_SEPARATOR.to_string(),
    };

    // Should not lint
    let _: &String = &MAIN_SEPARATOR.to_string();
    let _ = &MAIN_SEPARATOR.to_string();
    let _ = V {
        f: &MAIN_SEPARATOR.to_string(),
    };
}
