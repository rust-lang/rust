// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(deprecated_owned_vector)];

// Verify the compiler fails with an error on infinite function
// recursions.

struct Data(~Option<Data>);

fn generic<T>( _ : ~[(Data,T)] ) {
    //~^ ERROR reached the recursion limit during monomorphization
    let rec : ~[(Data,(bool,T))] = ~[];
    generic( rec );
}


fn main () {
    // Use generic<T> at least once to trigger instantiation.
    let input : ~[(Data,())] = ~[];
    generic(input);
}
