// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by the Rust Project Developers

// Copyright (c) 2013-2014 The Rust Project Developers
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of "The Computer Language Benchmarks Game" nor
//   the name of "The Computer Language Shootout Benchmarks" nor the
//   names of its contributors may be used to endorse or promote
//   products derived from this software without specific prior
//   written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

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
        FromStr::from_str(args[1].as_slice()).unwrap()
    };
    pidigits(n);
}
