//@ run-pass
#![allow(unused_mut)]
#![allow(unused_variables)]

#![feature(box_patterns)]

enum E {
    StructVar { boxed: Box<i32> }
}

fn main() {

    // Test matching each shorthand notation for field patterns.
    let mut a = E::StructVar { boxed: Box::new(3) };
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
    let mut a = E::StructVar { boxed: Box::new(3) };
    match a {
        E::StructVar { boxed: box ref mut num } => { }
    }
    match a {
        E::StructVar { boxed: ref mut num } => { }
    }

}
