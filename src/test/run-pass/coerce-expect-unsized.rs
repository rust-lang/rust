// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(unknown_features)]
#![feature(box_syntax)]

use std::fmt::Debug;

// Check that coercions apply at the pointer level and don't cause
// rvalue expressions to be unsized. See #20169 for more information.

pub fn main() {
    // FIXME #22405: We cannot infer the type `Box<[isize; k]>` for
    // the r-value expression from the context `Box<[isize]>`, and
    // therefore the `box EXPR` desugaring breaks down.
    //
    // One could reasonably claim that the `box EXPR` desugaring is
    // effectively regressing half of Issue #20169. Hopefully we will
    // eventually fix that, at which point the `Box::new` calls below
    // should be replaced wth uses of `box`.

    let _: Box<[isize]> = Box::new({ [1, 2, 3] });
    let _: Box<[isize]> = Box::new(if true { [1, 2, 3] } else { [1, 3, 4] });
    let _: Box<[isize]> = Box::new(match true { true => [1, 2, 3], false => [1, 3, 4] });
    let _: Box<Fn(isize) -> _> = Box::new({ |x| (x as u8) });
    let _: Box<Debug> = Box::new(if true { false } else { true });
    let _: Box<Debug> = Box::new(match true { true => 'a', false => 'b' });

    let _: &[isize] = &{ [1, 2, 3] };
    let _: &[isize] = &if true { [1, 2, 3] } else { [1, 3, 4] };
    let _: &[isize] = &match true { true => [1, 2, 3], false => [1, 3, 4] };
    let _: &Fn(isize) -> _ = &{ |x| (x as u8) };
    let _: &Debug = &if true { false } else { true };
    let _: &Debug = &match true { true => 'a', false => 'b' };

    let _: Box<[isize]> = Box::new([1, 2, 3]);
    let _: Box<Fn(isize) -> _> = Box::new(|x| (x as u8));

    let _: Vec<Box<Fn(isize) -> _>> = vec![
        Box::new(|x| (x as u8)),
        Box::new(|x| (x as i16 as u8)),
    ];
}
