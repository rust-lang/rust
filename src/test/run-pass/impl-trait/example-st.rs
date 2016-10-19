// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait)]

struct State;
type Error = ();

trait Bind<F> {
    type Output;
    fn bind(self, f: F) -> Self::Output;
}

fn bind<T, U, A, B, F>(mut a: A, mut f: F)
                       -> impl FnMut(&mut State) -> Result<U, Error>
where F: FnMut(T) -> B,
      A: FnMut(&mut State) -> Result<T, Error>,
      B: FnMut(&mut State) -> Result<U, Error>
{
    move |state | {
        let r = a(state)?;
        f(r)(state)
    }
}

fn atom<T>(x: T) -> impl FnMut(&mut State) -> Result<T, Error> {
    let mut x = Some(x);
    move |_| x.take().map_or(Err(()), Ok)
}

fn main() {
    assert_eq!(bind(atom(5), |x| atom(x > 4))(&mut State), Ok(true));
}
