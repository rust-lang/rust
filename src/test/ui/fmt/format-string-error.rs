// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-tab

fn main() {
    println!("{");
    //~^ ERROR invalid format string: expected `'}'` but string was terminated
    println!("{{}}");
    println!("}");
    //~^ ERROR invalid format string: unmatched `}` found
    let _ = format!("{_foo}", _foo = 6usize);
    //~^ ERROR invalid format string: invalid argument name `_foo`
    let _ = format!("{_}", _ = 6usize);
    //~^ ERROR invalid format string: invalid argument name `_`
    let _ = format!("{");
    //~^ ERROR invalid format string: expected `'}'` but string was terminated
    let _ = format!("}");
    //~^ ERROR invalid format string: unmatched `}` found
    let _ = format!("{\\}");
    //~^ ERROR invalid format string: expected `'}'`, found `'\\'`
    let _ = format!("\n\n\n{\n\n\n");
    //~^ ERROR invalid format string
    let _ = format!(r###"



	{"###);
    //~^ ERROR invalid format string
    let _ = format!(r###"



	{

"###);
    //~^^ ERROR invalid format string
    let _ = format!(r###"



	}

"###);
    //~^^^ ERROR invalid format string
    let _ = format!(r###"



        }

"###);
    //~^^^ ERROR invalid format string: unmatched `}` found
}
