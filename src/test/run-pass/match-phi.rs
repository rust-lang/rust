// pretty-expanded FIXME #23616

#![allow(dead_assignment)]
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
