// Issue #16205.

#![feature(box_syntax)]

struct Foo {
    a: [Box<isize>; 3],
}

fn main() {
    let mut y = 1;
    let x = Some(&mut y);
    for &a in x.iter() {    //~ ERROR cannot move out
    }

    let f = Foo {
        a: [box 3, box 4, box 5],
    };
    for &a in &f.a {  //~ ERROR cannot move out
    }

    let x: Option<Box<_>> = Some(box 1);
    for &a in x.iter() {    //~ ERROR cannot move out
    }
}
