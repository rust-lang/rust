// ignore-tidy-linelength
// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

#![feature(unboxed_closures)]

use std::io::Read;

fn to_fn_once<A,F:FnOnce<A>>(f: F) -> F { f }

fn main() {
    let x = 1;
    to_fn_once(move|| { x = 2; });
    //[ast]~^ ERROR: cannot assign to immutable captured outer variable
    //[mir]~^^ ERROR: cannot assign to `x`, as it is not declared as mutable

    let s = std::io::stdin();
    to_fn_once(move|| { s.read_to_end(&mut Vec::new()); });
    //[ast]~^ ERROR: cannot borrow immutable captured outer variable
    //[mir]~^^ ERROR: cannot borrow `s` as mutable, as it is not declared as mutable
}
