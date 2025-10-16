static mut a: Box<isize> = Box::new(3);
//~^ ERROR allocations are not allowed in statics [E0010]

fn main() {}
