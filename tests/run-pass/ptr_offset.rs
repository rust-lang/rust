fn f() -> i32 { 42 }

fn main() {
    let v = [1i16, 2];
    let x = &v as *const [i16; 2] as *const i16;
    let x = unsafe { x.offset(1) };
    assert_eq!(unsafe { *x }, 2);

    // fn ptr offset
    unsafe {
        let p = f as fn() -> i32 as usize;
        let x = (p as *mut u32).offset(0) as usize;
        let f: fn() -> i32 = std::mem::transmute(x);
        assert_eq!(f(), 42);
    }
}
