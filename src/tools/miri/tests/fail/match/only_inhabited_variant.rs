// rust-lang/miri#4778
#![feature(never_type)]

#[repr(C)]
#[allow(dead_code)]
enum E {
  V0, // discriminant: 0
  V1(!), // 1
}

fn main() {
    assert_eq!(std::mem::size_of::<E>(), 4);

    let val = 1u32;
    let ptr = (&raw const val).cast::<E>();
    let r = unsafe { &*ptr };
    match r { //~ ERROR: read discriminant of an uninhabited enum variant
        E::V0 => {}
        E::V1(_) => {}
    }
}
