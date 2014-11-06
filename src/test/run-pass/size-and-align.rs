// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum clam<T> { a(T, int), b, }

fn uhoh<T>(v: Vec<clam<T>> ) {
    match v[1] {
      clam::a::<T>(ref _t, ref u) => {
          println!("incorrect");
          println!("{}", u);
          panic!();
      }
      clam::b::<T> => { println!("correct"); }
    }
}

pub fn main() {
    let v: Vec<clam<int>> = vec!(clam::b::<int>, clam::b::<int>, clam::a::<int>(42, 17));
    uhoh::<int>(v);
}
