// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! parallel {
    (
        // If future has `pred`/`moelarry` fragments (where "pred" is
        // "like expr, but with `{` in its FOLLOW set"), then could
        // use `pred` instead of future-proof erroring here. See also:
        //
        // https://github.com/rust-lang/rfcs/pull/1384#issuecomment-160165525
        for $id:ident in $iter:expr { //~ WARN `$iter:expr` is followed by `{`
            $( $inner:expr; )*
        }
    ) => {};
}


fn main() {
    parallel! {
        for i in 0..n {
            x += i; //~ ERROR no rules expected the token `+=`
        }
    }
}
