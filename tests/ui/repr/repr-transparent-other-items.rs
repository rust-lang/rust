// See also repr-transparent.rs

#[repr(transparent)] //~ ERROR should be applied to a struct
fn cant_repr_this() {}

#[repr(transparent)] //~ ERROR should be applied to a struct
static CANT_REPR_THIS: u32 = 0;

fn main() {}
