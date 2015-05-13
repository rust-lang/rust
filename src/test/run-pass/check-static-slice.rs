// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the various ways of getting to a reference to a vec (both sized
// and unsized) work properly.


const AA: [isize; 3] = [1, 2, 3];
const AB: &'static [isize; 3] = &AA;
const AC: &'static [isize] = AB;
const AD: &'static [isize] = &AA;
const AE: &'static [isize; 3] = &[1, 2, 3];
const AF: &'static [isize] = &[1, 2, 3];

const CA: isize = AA[0];
const CB: isize = AB[1];
const CC: isize = AC[2];
const CD: isize = AD[0];
const CE: isize = AE[1];
const CF: isize = AF[2];

const AG: &'static isize = &AA[2];

fn main () {
    let b: &[isize] = &[1, 2, 3];
    assert!(AC == b);
    assert!(AD == b);
    assert!(AF == b);
    assert!(*AG == 3);

    assert!(CA == 1);
    assert!(CB == 2);
    assert!(CC == 3);
    assert!(CD == 1);
    assert!(CE == 2);
    assert!(CF == 3);
}
