// check-pass

use std::fmt;

// CoerceUnsized is not implemented for tuples. You can still create
// an unsized tuple by transmuting a trait object.
fn any<T>() -> T { unreachable!() }

fn main() {
    let t: &(u8, dyn fmt::Debug) = any();
    println!("{:?}", &t.1);
}
