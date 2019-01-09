// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

#![allow(unused_variables)]
#![allow(unused_assignments)]

fn separate_arms() {
    // Here both arms perform assignments, but only is illegal.

    let mut x = None;
    match x {
        None => {
            // It is ok to reassign x here, because there is in
            // fact no outstanding loan of x!
            x = Some(0);
        }
        Some(ref r) => {
            x = Some(1); //[ast]~ ERROR cannot assign
            //[mir]~^ ERROR cannot assign to `x` because it is borrowed
            drop(r);
        }
    }
    x.clone(); // just to prevent liveness warnings
}

fn main() {}
