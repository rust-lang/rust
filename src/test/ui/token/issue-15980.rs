// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;

fn main(){
    let x: io::IoResult<()> = Ok(());
    //~^ ERROR cannot find type `IoResult` in module `io`
    //~| NOTE did you mean `Result`?
    match x {
        Err(ref e) if e.kind == io::EndOfFile {
            //~^ NOTE while parsing this struct
            return
            //~^ ERROR expected identifier, found keyword `return`
            //~| NOTE expected identifier, found keyword
        }
        //~^ NOTE expected one of `.`, `=>`, `?`, or an operator here
        _ => {}
        //~^ ERROR expected one of `.`, `=>`, `?`, or an operator, found `_`
        //~| NOTE unexpected token
    }
}
