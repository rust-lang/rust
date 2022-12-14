// run-pass

#![allow(dead_code)]




// pretty-expanded FIXME #23616

struct Large {a: isize,
             b: isize,
             c: isize,
             d: isize,
             e: isize,
             f: isize,
             g: isize,
             h: isize,
             i: isize,
             j: isize,
             k: isize,
             l: isize}
fn f() {
    let _foo: Large =
        Large {a: 0,
         b: 0,
         c: 0,
         d: 0,
         e: 0,
         f: 0,
         g: 0,
         h: 0,
         i: 0,
         j: 0,
         k: 0,
         l: 0};
}

pub fn main() { f(); }
