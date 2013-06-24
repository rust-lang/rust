// xfail-fast #7103 `extern mod` does not work on windows
// xfail-pretty - does not converge

// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra; // {En,De}codable
mod submod {
    // if any of these are implemented without global calls for any
    // function calls, then being in a submodule will (correctly)
    // cause errors about unrecognised module `std` (or `extra`)
    #[deriving(Eq, Ord, TotalEq, TotalOrd,
               IterBytes,
               Clone, DeepClone,
               ToStr, Rand,
               Encodable, Decodable)]
    enum A { A1(uint), A2(int) }

    #[deriving(Eq, Ord, TotalEq, TotalOrd,
               IterBytes,
               Clone, DeepClone,
               ToStr, Rand,
               Encodable, Decodable)]
    struct B { x: uint, y: int }

    #[deriving(Eq, Ord, TotalEq, TotalOrd,
               IterBytes,
               Clone, DeepClone,
               ToStr, Rand,
               Encodable, Decodable)]
    struct C(uint, int);

}

fn main() {}
