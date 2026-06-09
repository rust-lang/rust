// See also repr-transparent.rs

#[repr(transparent)] //~ ERROR attribute cannot be used on
fn cant_repr_this() {}

#[repr(transparent)] //~ ERROR attribute cannot be used on
static CANT_REPR_THIS: u32 = 0;

fn main() {}
