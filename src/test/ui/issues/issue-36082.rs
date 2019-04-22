use std::cell::RefCell;

fn main() {
    let mut r = 0;
    let s = 0;
    let x = RefCell::new((&mut r,s));

    let val: &_ = x.borrow().0;
    //~^ ERROR temporary value dropped while borrowed [E0716]
    //~| NOTE temporary value is freed at the end of this statement
    //~| NOTE creates a temporary which is freed while still in use
    //~| NOTE consider using a `let` binding to create a longer lived value
    println!("{}", val);
    //~^ borrow later used here
}
