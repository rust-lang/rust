// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn get_number() -> usize {
    10
}

fn get_reference(n: &usize) -> &usize {
    n
}

#[allow(clippy::many_single_char_names, clippy::double_parens)]
#[allow(unused_variables)]
#[warn(clippy::deref_addrof)]
fn main() {
    let a = 10;
    let aref = &a;

    let b = *&a;

    let b = *&get_number();

    let b = *get_reference(&a);

    let bytes: Vec<usize> = vec![1, 2, 3, 4];
    let b = *&bytes[1..2][0];

    //This produces a suggestion of 'let b = (a);' which
    //will trigger the 'unused_parens' lint
    let b = *&(a);

    let b = *(&a);

    #[rustfmt::skip]
    let b = *((&a));

    let b = *&&a;

    let b = **&aref;

    //This produces a suggestion of 'let b = *&a;' which
    //will trigger the 'clippy::deref_addrof' lint again
    let b = **&&a;

    {
        let mut x = 10;
        let y = *&mut x;
    }

    {
        //This produces a suggestion of 'let y = *&mut x' which
        //will trigger the 'clippy::deref_addrof' lint again
        let mut x = 10;
        let y = **&mut &mut x;
    }
}
