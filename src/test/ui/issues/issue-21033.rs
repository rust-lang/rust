// run-pass
#![allow(unused_mut)]
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

#![feature(box_patterns)]
#![feature(box_syntax)]

enum E {
    StructVar { boxed: Box<i32> }
}

fn main() {

    // Test matching each shorthand notation for field patterns.
    let mut a = E::StructVar { boxed: box 3 };
    match a {
        E::StructVar { box boxed } => { }
    }
    match a {
        E::StructVar { box ref boxed } => { }
    }
    match a {
        E::StructVar { box mut boxed } => { }
    }
    match a {
        E::StructVar { box ref mut boxed } => { }
    }
    match a {
        E::StructVar { ref boxed } => { }
    }
    match a {
        E::StructVar { ref mut boxed } => { }
    }
    match a {
        E::StructVar { mut boxed } => { }
    }

    // Test matching non shorthand notation. Recreate a since last test
    // moved `boxed`
    let mut a = E::StructVar { boxed: box 3 };
    match a {
        E::StructVar { boxed: box ref mut num } => { }
    }
    match a {
        E::StructVar { boxed: ref mut num } => { }
    }

}
