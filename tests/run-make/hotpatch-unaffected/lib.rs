// hotpatch has two requirements:
// 1. the first instruction of a functin must be at least two bytes long
// 2. there must not be a jump to the first instruction

// the functions in this file already fulfill the conditions so hotpatch should not affect them

// --------------------------------------------------------------------------------------------

#[no_mangle]
#[inline(never)]
pub fn return_42() -> i32 {
    42
}

// --------------------------------------------------------------------------------------------
// This tailcall does not jump to the first instruction so hotpatch should leave it unaffected

#[no_mangle]
pub fn tailcall(a: i32) -> i32 {
    if a > 10000 {
        return a;
    }

    if a % 2 == 0 { tailcall(a / 2) } else { tailcall(a * 3 + 1) }
}
