// A variant of raw-ptr-const that directly constructs a raw pointer.

const CONST_RAW: *const Vec<i32> = std::ptr::addr_of!(Vec::new());
//~^ ERROR cannot take address of a temporary

fn main() {}
