// run-pass
#![feature(const_extern_fn)]

const extern "C" fn foo1(val: u8) -> u8 {
    val + 1
}

const extern "C" fn foo2(val: u8) -> u8 {
    val + 1
}

const unsafe extern "C" fn bar1(val: bool) -> bool {
    !val
}

const unsafe extern "C" fn bar2(val: bool) -> bool {
    !val
}


fn main() {
    let a: [u8; foo1(25) as usize] = [0; 26];
    let b: [u8; foo2(25) as usize] = [0; 26];
    assert_eq!(a, b);

    let bar1_res = unsafe { bar1(false) };
    let bar2_res = unsafe { bar2(false) };
    assert!(bar1_res);
    assert_eq!(bar1_res, bar2_res);

    let _foo1_cast: extern "C" fn(u8) -> u8 = foo1;
    let _foo2_cast: extern "C" fn(u8) -> u8 = foo2;
    let _bar1_cast: unsafe extern "C" fn(bool) -> bool = bar1;
    let _bar2_cast: unsafe extern "C" fn(bool) -> bool = bar2;
}
