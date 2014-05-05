// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate num;

use std::from_str::FromStr;
use std::num::One;
use std::num::Zero;
use std::num::FromPrimitive;
use num::Integer;
use num::bigint::BigInt;

struct Context {
    numer: BigInt,
    accum: BigInt,
    denom: BigInt,
}

impl Context {
    fn new() -> Context {
        Context {
            numer: One::one(),
            accum: Zero::zero(),
            denom: One::one(),
        }
    }

    fn from_int(i: int) -> BigInt {
        FromPrimitive::from_int(i).unwrap()
    }

    fn extract_digit(&self) -> int {
        if self.numer > self.accum {return -1;}
        let (q, r) =
            (self.numer * Context::from_int(3) + self.accum)
            .div_rem(&self.denom);
        if r + self.numer >= self.denom {return -1;}
        q.to_int().unwrap()
    }

    fn next_term(&mut self, k: int) {
        let y2 = Context::from_int(k * 2 + 1);
        self.accum = (self.accum + (self.numer << 1)) * y2;
        self.numer = self.numer * Context::from_int(k);
        self.denom = self.denom * y2;
    }

    fn eliminate_digit(&mut self, d: int) {
        let d = Context::from_int(d);
        let ten = Context::from_int(10);
        self.accum = (self.accum - self.denom * d) * ten;
        self.numer = self.numer * ten;
    }
}

fn pidigits(n: int) {
    let mut k = 0;
    let mut context = Context::new();

    for i in range(1, n + 1) {
        let mut d;
        loop {
            k += 1;
            context.next_term(k);
            d = context.extract_digit();
            if d != -1 {break;}
        }

        print!("{}", d);
        if i % 10 == 0 {print!("\t:{}\n", i);}

        context.eliminate_digit(d);
    }

    let m = n % 10;
    if m != 0 {
        for _ in range(m, 10) { print!(" "); }
        print!("\t:{}\n", n);
    }
}

fn main() {
    let args = std::os::args();
    let args = args.as_slice();
    let n = if args.len() < 2 {
        512
    } else {
        FromStr::from_str(args[1]).unwrap()
    };
    pidigits(n);
}
