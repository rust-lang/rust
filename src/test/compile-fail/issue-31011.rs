// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! log {
    ( $ctx:expr, $( $args:expr),* ) => {
        if $ctx.trace {
        //~^ no field `trace` on type `&T`
            println!( $( $args, )* );
        }
    }
}

// Create a structure.
struct Foo {
  trace: bool,
}

// Generic wrapper calls log! with a structure.
fn wrap<T>(context: &T) -> ()
{
    log!(context, "entered wrapper");
    //~^ in this expansion of log!
}

fn main() {
    // Create a structure.
    let x = Foo { trace: true };
    log!(x, "run started");
    // Apply a closure which accesses internal fields.
    wrap(&x);
    log!(x, "run finished");
}
