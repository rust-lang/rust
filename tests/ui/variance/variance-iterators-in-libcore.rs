//@ run-pass

#![allow(dead_code)]

use std::iter::{Fuse, Zip};

fn fuse_covariant<'a, I>(iter: Fuse<&'static I>) -> Fuse<&'a I> { iter }
fn zip_covariant<'a, A, B>(iter: Zip<&'static A, &'static B>) -> Zip<&'a A, &'a B> { iter }

fn main() { }
