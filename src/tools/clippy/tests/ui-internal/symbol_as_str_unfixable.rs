//@no-rustfix: paths that don't exist yet
#![feature(rustc_private)]

extern crate rustc_span;

use rustc_span::Symbol;

fn f(s: Symbol) {
    s.as_str() == "xyz123";
    //~^ symbol_as_str
    s.as_str() == "with-dash";
    //~^ symbol_as_str
    s.as_str() == "with.dot";
    //~^ symbol_as_str
}
