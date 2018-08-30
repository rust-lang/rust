// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




trait to_str {
    fn to_string_(&self) -> String;
}
impl to_str for isize {
    fn to_string_(&self) -> String { self.to_string() }
}
impl to_str for String {
    fn to_string_(&self) -> String { self.clone() }
}
impl to_str for () {
    fn to_string_(&self) -> String { "()".to_string() }
}

trait map<T> {
    fn map<U, F>(&self, f: F) -> Vec<U> where F: FnMut(&T) -> U;
}
impl<T> map<T> for Vec<T> {
    fn map<U, F>(&self, mut f: F) -> Vec<U> where F: FnMut(&T) -> U {
        let mut r = Vec::new();
        for i in self {
            r.push(f(i));
        }
        r
    }
}

fn foo<U, T: map<U>>(x: T) -> Vec<String> {
    x.map(|_e| "hi".to_string() )
}
fn bar<U:to_str,T:map<U>>(x: T) -> Vec<String> {
    x.map(|_e| _e.to_string_() )
}

pub fn main() {
    assert_eq!(foo(vec![1]), ["hi".to_string()]);
    assert_eq!(bar::<isize, Vec<isize> >(vec![4, 5]), ["4".to_string(), "5".to_string()]);
    assert_eq!(bar::<String, Vec<String> >(vec!["x".to_string(), "y".to_string()]),
               ["x".to_string(), "y".to_string()]);
    assert_eq!(bar::<(), Vec<()>>(vec![()]), ["()".to_string()]);
}
