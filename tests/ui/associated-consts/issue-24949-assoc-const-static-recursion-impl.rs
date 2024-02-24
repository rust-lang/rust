// Check for recursion involving references to impl-associated const.

trait Foo {
    const BAR: u32;
}

const IMPL_REF_BAR: u32 = GlobalImplRef::BAR; //~ ERROR E0391

struct GlobalImplRef;

impl GlobalImplRef {
    const BAR: u32 = IMPL_REF_BAR;
}

fn main() {}
