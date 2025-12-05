use std::mem;

#[repr(C, usize)]
#[allow(unused)]
enum E {
    Var1(usize),
    Var2(usize),
}

// Doing a copy at integer type should lose provenance.
// This tests the case where provenacne is hiding in the discriminant of an enum.
fn main() {
    assert_eq!(mem::size_of::<E>(), 2 * mem::size_of::<usize>());

    // We want to store provenance in the enum discriminant, but the value still needs to
    // be valid atfor the type. So we split provenance and data.
    let ptr = &42;
    let ptr = ptr as *const i32;
    let ptrs = [(ptr.with_addr(0), ptr)];
    // Typed copy at the enum type.
    let ints: [E; 1] = unsafe { mem::transmute(ptrs) };
    // Read the discriminant.
    let discr = unsafe { (&raw const ints[0]).cast::<*const i32>().read() };
    // Take the provenance from there, together with the original address.
    let ptr = discr.with_addr(ptr.addr());
    // There should be no provenance is `discr`, so this should be UB.
    let _val = unsafe { *ptr }; //~ERROR: dangling
}
