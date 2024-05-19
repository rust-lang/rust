//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-permissive-provenance
use std::mem;
use std::ptr;

fn eq_ref<T>(x: &T, y: &T) -> bool {
    x as *const _ == y as *const _
}

fn f() -> i32 {
    42
}

fn ptr_int_casts() {
    // int-ptr-int
    assert_eq!(1 as *const i32 as usize, 1);
    assert_eq!((1 as *const i32).wrapping_offset(4) as usize, 1 + 4 * 4);

    // negative overflowing wrapping_offset (going through memory because
    // this used to trigger an ICE on 32bit)
    let val = &mut ptr::null();
    *val = (1 as *const u8).wrapping_offset(-4);
    assert_eq!(*val as usize, usize::MAX - 2);

    // ptr-int-ptr
    {
        let x = 13;
        let mut y = &x as &_ as *const _ as usize;
        y += 13;
        y -= 13;
        let y = y as *const _;
        assert!(eq_ref(&x, unsafe { &*y }));
    }

    // fnptr-int-fnptr
    {
        let x: fn() -> i32 = f;
        let y: *mut u8 = unsafe { mem::transmute(x as fn() -> i32) };
        let mut y = y as usize;
        y += 13;
        y -= 13;
        let x: fn() -> i32 = unsafe { mem::transmute(y as *mut u8) };
        assert_eq!(x(), 42);
    }

    // involving types other than usize
    assert_eq!((-1i32) as usize as *const i32 as usize, (-1i32) as usize);
}

fn ptr_int_ops() {
    let v = [1i16, 2];
    let x = &v[1] as *const i16 as usize;
    // arithmetic
    let _y = x + 4;
    let _y = 4 + x;
    let _y = x - 2;
    // bit-operations, covered by alignment
    assert_eq!(x & 1, 0);
    assert_eq!(x & 0, 0);
    assert_eq!(1 & (x + 1), 1);
    let _y = !1 & x;
    let _y = !0 & x;
    let _y = x & !1;
    // remainder, covered by alignment
    assert_eq!(x % 2, 0);
    assert_eq!((x + 1) % 2, 1);
    // remainder with 1 is always 0
    assert_eq!(x % 1, 0);
}

fn main() {
    ptr_int_casts();
    ptr_int_ops();
}
