// run-rustfix
#![allow(unused, dead_code)]
fn first<T, 'a, 'b>() {} //~ ERROR incorrect parameter order
fn second<'a, T, 'b>() {} //~ ERROR incorrect parameter order
fn third<T, U, 'a>() {} //~ ERROR incorrect parameter order
fn fourth<'a, T, 'b, U, 'c, V>() {} //~ ERROR incorrect parameter order

fn main() {}
