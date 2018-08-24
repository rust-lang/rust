// http://rust-lang.org/COPYRIGHT.
//

// revisions: ast cmp
//[cmp]compile-flags: -Z borrowck=compare

#![feature(slice_patterns)]

fn main() {
    let mut a = [1, 2, 3, 4];
    let t = match a {
        [1, 2, ref tail..] => tail,
        _ => unreachable!()
    };
    println!("t[0]: {}", t[0]);
    a[2] = 0; //[ast]~ ERROR cannot assign to `a[..]` because it is borrowed
              //[cmp]~^ ERROR cannot assign to `a[..]` because it is borrowed (Ast)
              //[cmp]~| ERROR cannot assign to `a[..]` because it is borrowed (Mir)
    println!("t[0]: {}", t[0]);
    t[0];
}
