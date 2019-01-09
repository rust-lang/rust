#![allow(warnings)]

use std::iter::Zip;

fn zip_covariant<'a, A, B>(iter: Zip<&'static A, &'static B>) -> Zip<&'a A, &'a B> { iter }

fn main() { }
