// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

use core::prelude::*;

use core::iterator::*;
use core::vec;
use core::f64;
use core::cmp;
use core::num;
use sort;

// NB: this can probably be rewritten in terms of num::Num
// to be less f64-specific.

pub trait Stats {
    fn sum(self) -> f64;
    fn min(self) -> f64;
    fn max(self) -> f64;
    fn mean(self) -> f64;
    fn median(self) -> f64;
    fn var(self) -> f64;
    fn std_dev(self) -> f64;
    fn std_dev_pct(self) -> f64;
    fn median_abs_dev(self) -> f64;
    fn median_abs_dev_pct(self) -> f64;
}

impl<'self> Stats for &'self [f64] {
    fn sum(self) -> f64 {
        self.iter().fold(0.0, |p,q| p + *q)
    }

    fn min(self) -> f64 {
        assert!(self.len() != 0);
        self.iter().fold(self[0], |p,q| cmp::min(p, *q))
    }

    fn max(self) -> f64 {
        assert!(self.len() != 0);
        self.iter().fold(self[0], |p,q| cmp::max(p, *q))
    }

    fn mean(self) -> f64 {
        assert!(self.len() != 0);
        self.sum() / (self.len() as f64)
    }

    fn median(self) -> f64 {
        assert!(self.len() != 0);
        let mut tmp = vec::to_owned(self);
        sort::tim_sort(tmp);
        if tmp.len() & 1 == 0 {
            let m = tmp.len() / 2;
            (tmp[m] + tmp[m-1]) / 2.0
        } else {
            tmp[tmp.len() / 2]
        }
    }

    fn var(self) -> f64 {
        if self.len() == 0 {
            0.0
        } else {
            let mean = self.mean();
            let mut v = 0.0;
            for self.each |s| {
                let x = *s - mean;
                v += x*x;
            }
            v/(self.len() as f64)
        }
    }

    fn std_dev(self) -> f64 {
        f64::sqrt(self.var())
    }

    fn std_dev_pct(self) -> f64 {
        (self.std_dev() / self.mean()) * 100.0
    }

    fn median_abs_dev(self) -> f64 {
        let med = self.median();
        let abs_devs = self.map(|v| num::abs(med - *v));
        abs_devs.median()
    }

    fn median_abs_dev_pct(self) -> f64 {
        (self.median_abs_dev() / self.median()) * 100.0
    }
}
