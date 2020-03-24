// run-pass

#![feature(core_intrinsics)]
#![feature(const_fn)]
#![feature(const_if_match)]
#![feature(const_unreachable_unchecked)]

const unsafe fn f_std_hint_unreachable(x: u8) -> u8 {
    match x {
        42 => 34,
        _ => std::hint::unreachable_unchecked(),
    }
}

const unsafe fn f_std_intrinsics_unreachable(x: u8) -> u8 {
    match x {
        17 => 22,
        _ => std::intrinsics::unreachable(),
    }
}

const FOO:u8 = unsafe { f_std_hint_unreachable(42) };
const BAR:u8 = unsafe { f_std_intrinsics_unreachable(17) };

fn main() {
    assert_eq!(FOO, 34);
    assert_eq!(BAR, 22);
}
