// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that shapes respect linearize_ty_params().

enum option<T> {
    none,
    some(T),
}

type smallintmap<T> = @{mut v: ~[mut option<T>]};

fn mk<T>() -> smallintmap<T> {
    let v: ~[mut option<T>] = ~[mut];
    return @{mut v: move v};
}

fn f<T,U>() {
    let sim = mk::<U>();
    log(error, sim);
}

fn main() {
    f::<int,int>();
}

