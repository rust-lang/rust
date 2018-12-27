// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

use std::cell::RefCell;

fn main() {
    let mut r = 0;
    let s = 0;
    let x = RefCell::new((&mut r,s));

    let val: &_ = x.borrow().0;
    //[ast]~^ ERROR borrowed value does not live long enough [E0597]
    //[ast]~| NOTE temporary value dropped here while still borrowed
    //[ast]~| NOTE temporary value does not live long enough
    //[ast]~| NOTE consider using a `let` binding to increase its lifetime
    //[mir]~^^^^^ ERROR temporary value dropped while borrowed [E0716]
    //[mir]~| NOTE temporary value is freed at the end of this statement
    //[mir]~| NOTE creates a temporary which is freed while still in use
    //[mir]~| NOTE consider using a `let` binding to create a longer lived value
    println!("{}", val);
    //[mir]~^ borrow later used here
}
//[ast]~^ NOTE temporary value needs to live until here
