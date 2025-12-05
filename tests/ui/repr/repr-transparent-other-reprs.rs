// See also repr-transparent.rs

#[repr(transparent, C)] //~ ERROR cannot have other repr
struct TransparentPlusC {
    ptr: *const u8
}

#[repr(transparent, packed)] //~ ERROR cannot have other repr
struct TransparentPlusPacked(*const u8);

#[repr(transparent, align(2))] //~ ERROR cannot have other repr
struct TransparentPlusAlign(u8);

#[repr(transparent)] //~ ERROR cannot have other repr
#[repr(C)]
struct SeparateAttributes(*mut u8);

fn main() {}
