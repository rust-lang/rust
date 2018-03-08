// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn gratitude() {
    let for_you = false;
    if not for_you {
        //~^ ERROR expected `{`
        println!("I couldn't");
        //~^ ERROR expected one of
    }
}

fn qualification() {
    let the_worst = true;
    while not the_worst {
        //~^ ERROR expected one of
        println!("still pretty bad");
    }
}

fn defer() {
    let department = false;
    // `not` as one segment of a longer path doesn't trigger the smart help
    if not::my department {
        //~^ ERROR expected `{`
        println!("pass");
        //~^ ERROR expected one of
    }
}

fn should_we() {
    let not = true;
    if not  // lack of braces is [sic]
        println!("Then when?");
    //~^ ERROR expected `{`
}

fn main() {}
