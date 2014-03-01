// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Clone)]
enum Noun
{
    Atom(int),
    Cell(~Noun, ~Noun)
}

fn fas(n: &Noun) -> Noun
{
    match n
    {
        &Cell(~Atom(2), ~Cell(ref a, _)) => (**a).clone(),
        _ => fail!("Invalid fas pattern")
    }
}

pub fn main() {
    fas(&Cell(~Atom(2), ~Cell(~Atom(2), ~Atom(3))));
}
