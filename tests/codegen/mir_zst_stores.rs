//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
use std::marker::PhantomData;

#[derive(Copy, Clone)]
struct Zst {
    phantom: PhantomData<Zst>,
}

// CHECK-LABEL: @mir
// CHECK-NOT: store{{.*}}undef
#[no_mangle]
pub fn mir() {
    let x = Zst { phantom: PhantomData };
    let y = (x, 0);
    drop(y);
    drop((0, x));
}
