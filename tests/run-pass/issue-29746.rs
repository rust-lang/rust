// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// zip!(a1,a2,a3,a4) is equivalent to:
//  a1.zip(a2).zip(a3).zip(a4).map(|(((x1,x2),x3),x4)| (x1,x2,x3,x4))
macro_rules! zip {
    // Entry point
    ([$a:expr, $b:expr, $($rest:expr),*]) => {
        zip!([$($rest),*], $a.zip($b), (x,y), [x,y])
    };

    // Intermediate steps to build the zipped expression, the match pattern, and
    //  and the output tuple of the closure, using macro hygene to repeatedly
    //  introduce new variables named 'x'.
    ([$a:expr, $($rest:expr),*], $zip:expr, $pat:pat, [$($flat:expr),*]) => {
        zip!([$($rest),*], $zip.zip($a), ($pat,x), [$($flat),*, x])
    };

    // Final step
    ([], $zip:expr, $pat:pat, [$($flat:expr),+]) => {
        $zip.map(|$pat| ($($flat),+))
    };

    // Comma
    ([$a:expr], $zip:expr, $pat:pat, [$($flat:expr),*]) => {
        zip!([$a,], $zip, $pat, [$($flat),*])
    };
}

fn main() {
    let p1 = vec![1i32,    2].into_iter();
    let p2 = vec!["10",    "20"].into_iter();
    let p3 = vec![100u16,  200].into_iter();
    let p4 = vec![1000i64, 2000].into_iter();

    let e = zip!([p1,p2,p3,p4]).collect::<Vec<_>>();
    assert_eq!(e[0], (1i32,"10",100u16,1000i64));
}
