// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub enum Expr<'var, VAR> {
    Let(Box<Expr<'var, VAR>>,
        Box<for<'v: 'var> Fn(Expr<'v, VAR>) -> Expr<'v, VAR> + 'var>)
}

pub fn add<'var, VAR>
                      (a: Expr<'var, VAR>, b: Expr<'var, VAR>) -> Expr<'var, VAR> {
    loop {}
}

pub fn let_<'var, VAR, F: for<'v: 'var> Fn(Expr<'v, VAR>) -> Expr<'v, VAR>>
                       (a: Expr<'var, VAR>, b: F) -> Expr<'var, VAR> {
    loop {}
}

fn main() {
    let ex = |x| {
        let_(add(x,x), |y| { //~ ERROR unable to infer enough type information about `VAR`
            let_(add(x, x), |x|x)})};
}
