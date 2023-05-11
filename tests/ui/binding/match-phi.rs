// run-pass
#![allow(dead_code)]
#![allow(unused_assignments)]
// pretty-expanded FIXME #23616
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

enum thing { a, b, c, }

fn foo<F>(it: F) where F: FnOnce(isize) { it(10); }

pub fn main() {
    let mut x = true;
    match thing::a {
      thing::a => { x = true; foo(|_i| { } ) }
      thing::b => { x = false; }
      thing::c => { x = false; }
    }
}
