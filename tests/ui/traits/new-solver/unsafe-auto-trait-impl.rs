// compile-flags: -Ztrait-solver=next
// check-pass

struct Foo(*mut ());

unsafe impl Sync for Foo {}

fn main() {}
