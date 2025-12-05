//@ edition: 2015
//@ check-pass
#![allow(bare_trait_objects)]

pub struct FormatWith<'a, I, F> {
    sep: &'a str,
    /// FormatWith uses interior mutability because Display::fmt takes &self.
    inner: RefCell<Option<(I, F)>>,
}

use std::cell::RefCell;
use std::fmt;

struct Layout;

pub fn new_format<'a, I, F>(iter: I, separator: &'a str, f: F) -> FormatWith<'a, I, F>
where
    I: Iterator,
    F: FnMut(I::Item, &mut FnMut(&fmt::Display) -> fmt::Result) -> fmt::Result,
{
    FormatWith { sep: separator, inner: RefCell::new(Some((iter, f))) }
}

fn main() {
    let _ = new_format(0..32, " | ", |i, f| f(&format_args!("0x{:x}", i)));
}
