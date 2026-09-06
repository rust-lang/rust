// Verifies that functions with bool and char argument types can be called
// through function pointers with u8 and u32 argument types when compiling with
// -Zsanitizer-cfi-normalize-integers.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Zsanitizer-cfi-normalize-integers -Cunsafe-allow-abi-mismatch=sanitizer,sanitizer-cfi-normalize-integers
//@ run-pass

fn foo1(_: bool) -> i32 {
    1
}

fn foo2(_: char) -> i32 {
    2
}

fn foo3(_: isize) -> i32 {
    3
}

fn foo4(_: usize) -> i32 {
    4
}

#[cfg(target_pointer_width = "16")]
type Isize = i16;
#[cfg(target_pointer_width = "32")]
type Isize = i32;
#[cfg(target_pointer_width = "64")]
type Isize = i64;

#[cfg(target_pointer_width = "16")]
type Usize = u16;
#[cfg(target_pointer_width = "32")]
type Usize = u32;
#[cfg(target_pointer_width = "64")]
type Usize = u64;

fn main() {
    // bool is normalized to u8 and char to u32
    let f: fn(bool) -> i32 = std::hint::black_box(foo1);
    let f: fn(u8) -> i32 = std::hint::black_box(unsafe { std::mem::transmute(f) });
    assert_eq!(f(true as u8), 1);
    let f: fn(char) -> i32 = std::hint::black_box(foo2);
    let f: fn(u32) -> i32 = std::hint::black_box(unsafe { std::mem::transmute(f) });
    assert_eq!(f('a' as u32), 2);

    // isize and usize are normalized to the integer of the target pointer width
    let f: fn(isize) -> i32 = std::hint::black_box(foo3);
    let f: fn(Isize) -> i32 = std::hint::black_box(unsafe { std::mem::transmute(f) });
    assert_eq!(f(3), 3);
    let f: fn(usize) -> i32 = std::hint::black_box(foo4);
    let f: fn(Usize) -> i32 = std::hint::black_box(unsafe { std::mem::transmute(f) });
    assert_eq!(f(4), 4);
}
