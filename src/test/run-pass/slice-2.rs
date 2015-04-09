// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test slicing expressions on slices and Vecs.


fn main() {
    let x: &[isize] = &[1, 2, 3, 4, 5];
    let cmp: &[isize] = &[1, 2, 3, 4, 5];
    assert!(&x[..] == cmp);
    let cmp: &[isize] = &[3, 4, 5];
    assert!(&x[2..] == cmp);
    let cmp: &[isize] = &[1, 2, 3];
    assert!(&x[..3] == cmp);
    let cmp: &[isize] = &[2, 3, 4];
    assert!(&x[1..4] == cmp);

    let x: Vec<isize> = vec![1, 2, 3, 4, 5];
    let cmp: &[isize] = &[1, 2, 3, 4, 5];
    assert!(&x[..] == cmp);
    let cmp: &[isize] = &[3, 4, 5];
    assert!(&x[2..] == cmp);
    let cmp: &[isize] = &[1, 2, 3];
    assert!(&x[..3] == cmp);
    let cmp: &[isize] = &[2, 3, 4];
    assert!(&x[1..4] == cmp);

    let x: &mut [isize] = &mut [1, 2, 3, 4, 5];
    {
        let cmp: &mut [isize] = &mut [1, 2, 3, 4, 5];
        assert!(&mut x[..] == cmp);
    }
    {
        let cmp: &mut [isize] = &mut [3, 4, 5];
        assert!(&mut x[2..] == cmp);
    }
    {
        let cmp: &mut [isize] = &mut [1, 2, 3];
        assert!(&mut x[..3] == cmp);
    }
    {
        let cmp: &mut [isize] = &mut [2, 3, 4];
        assert!(&mut x[1..4] == cmp);
    }

    let mut x: Vec<isize> = vec![1, 2, 3, 4, 5];
    {
        let cmp: &mut [isize] = &mut [1, 2, 3, 4, 5];
        assert!(&mut x[..] == cmp);
    }
    {
        let cmp: &mut [isize] = &mut [3, 4, 5];
        assert!(&mut x[2..] == cmp);
    }
    {
        let cmp: &mut [isize] = &mut [1, 2, 3];
        assert!(&mut x[..3] == cmp);
    }
    {
        let cmp: &mut [isize] = &mut [2, 3, 4];
        assert!(&mut x[1..4] == cmp);
    }
}
