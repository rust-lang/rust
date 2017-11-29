// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
// compile-flags: --error-format=human

// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

#![feature(default_type_parameter_fallback)]

use std::vec::IntoIter;
use std::iter::Sum;
use std::slice::Iter;
use std::ops::Add;

trait Itarator: Iterator {
    type Iten;
    // Bug: Even though it's unambiguos, using Self::Iten dosen't work here.
    // probably can be fixed in fn associated_path_def_to_ty.
    fn foo<T:Default=<Self as Itarator>::Iten>(&self) -> T {
        T::default()
    }

    fn suma<S=<<Self as Itarator>::Iten as Add>::Output>(self) -> S
        where Self: Sized,
              S: Sum<<Self as Iterator>::Item>,
    {
        Sum::sum(self)
    }
}

impl Itarator for IntoIter<u32> {
    type Iten = <IntoIter<u32> as Iterator>::Item;
}

impl<'a> Itarator for Iter<'a, u32> {
    type Iten = <Iter<'a, u32> as Iterator>::Item;
}

fn main() {
    let x = vec![0u32];
    {
        let v = x.iter();
        // Bug: if we put a cast such as `as u64`, inference fails.
        //The usual guess is that we propagate the origin but not the default of the inference var.
        v.suma();
    }
    x.clone().into_iter().suma();
    x.into_iter().suma();
}
