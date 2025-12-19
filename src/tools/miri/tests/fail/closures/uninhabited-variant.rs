// Motivated by rust-lang/rust#138961, this shows how invalid discriminants interact with
// closure captures.
#![feature(never_type)]

#[repr(C)]
#[allow(dead_code)]
enum E {
    V0,    // discriminant: 0
    V1,    // 1
    V2(!), // 2
}

fn main() {
    assert_eq!(std::mem::size_of::<E>(), 4);

    let val = 2u32;
    let ptr = (&raw const val).cast::<E>();
    let r = unsafe { &*ptr };
    let f = || {
        // After rust-lang/rust#138961, constructing the closure performs a reborrow of r.
        // Nevertheless, the discriminant is only actually inspected when the closure
        // is called.
        match r {
            //~^ ERROR: read discriminant of an uninhabited enum variant
            E::V0 => {}
            E::V1 => {}
            E::V2(_) => {}
        }
    };

    f();
}
