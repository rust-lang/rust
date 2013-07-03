// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

use std::int;

trait vec_monad<A> {
    fn bind<B:Copy>(&self, f: &fn(&A) -> ~[B]) -> ~[B];
}

impl<A> vec_monad<A> for ~[A] {
    fn bind<B:Copy>(&self, f: &fn(&A) -> ~[B]) -> ~[B] {
        let mut r = ~[];
        for self.iter().advance |elt| {
            r.push_all_move(f(elt));
        }
        r
    }
}

trait option_monad<A> {
    fn bind<B>(&self, f: &fn(&A) -> Option<B>) -> Option<B>;
}

impl<A> option_monad<A> for Option<A> {
    fn bind<B>(&self, f: &fn(&A) -> Option<B>) -> Option<B> {
        match *self {
            Some(ref a) => { f(a) }
            None => { None }
        }
    }
}

fn transform(x: Option<int>) -> Option<~str> {
    x.bind(|n| Some(*n + 1) ).bind(|n| Some(int::to_str(*n)) )
}

pub fn main() {
    assert_eq!(transform(Some(10)), Some(~"11"));
    assert_eq!(transform(None), None);
    assert!((~[~"hi"])
        .bind(|x| ~[x.clone(), *x + ~"!"] )
        .bind(|x| ~[x.clone(), *x + ~"?"] ) ==
        ~[~"hi", ~"hi?", ~"hi!", ~"hi!?"]);
}
