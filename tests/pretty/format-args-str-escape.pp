#![feature(prelude_import)]
#![no_std]
#[macro_use]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:format-args-str-escape.pp

fn main() {
    { ::std::io::_print(format_args!("\u{1b}[1mHello, world!\u{1b}[0m\n")); };
    { ::std::io::_print(format_args!("\u{1b}[1mHello, world!\u{1b}[0m\n")); };
    {
        ::std::io::_print(format_args!("Not an escape sequence: \\u{{1B}}[1mbold\\x1B[0m\n"));
    };
    {
        ::std::io::_print(format_args!("{0}\n",
                "\x1B[1mHello, world!\x1B[0m"));
    };
}
