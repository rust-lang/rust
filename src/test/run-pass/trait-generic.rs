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
    fn to_string(&self) -> String;
}
impl to_str for int {
    fn to_string(&self) -> String { self.to_str().to_strbuf() }
}
impl to_str for String {
    fn to_string(&self) -> String { self.clone() }
}
impl to_str for () {
    fn to_string(&self) -> String { "()".to_strbuf() }
}

trait map<T> {
    fn map<U>(&self, f: |&T| -> U) -> Vec<U> ;
}
impl<T> map<T> for Vec<T> {
    fn map<U>(&self, f: |&T| -> U) -> Vec<U> {
        let mut r = Vec::new();
        // FIXME: #7355 generates bad code with VecIterator
        for i in range(0u, self.len()) {
            r.push(f(self.get(i)));
        }
        r
    }
}

fn foo<U, T: map<U>>(x: T) -> Vec<String> {
    x.map(|_e| "hi".to_strbuf() )
}
fn bar<U:to_str,T:map<U>>(x: T) -> Vec<String> {
    x.map(|_e| _e.to_string() )
}

pub fn main() {
    assert_eq!(foo(vec!(1)), vec!("hi".to_strbuf()));
    assert_eq!(bar::<int, Vec<int> >(vec!(4, 5)), vec!("4".to_strbuf(), "5".to_strbuf()));
    assert_eq!(bar::<String, Vec<String> >(vec!("x".to_strbuf(), "y".to_strbuf())),
               vec!("x".to_strbuf(), "y".to_strbuf()));
    assert_eq!(bar::<(), Vec<()>>(vec!(())), vec!("()".to_strbuf()));
}
