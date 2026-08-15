static mut a: Box<isize> = Box::new(3);
//~^ ERROR cannot call non-const associated function

fn main() {}
