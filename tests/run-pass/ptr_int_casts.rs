fn eq_ref<T>(x: &T, y: &T) -> bool {
    x as *const _ == y as *const _
}

fn main() {
    // int-ptr-int
    assert_eq!(1 as *const i32 as usize, 1);

    {   // ptr-int-ptr
        let x = 13;
        let y = &x as *const _ as usize;
        let y = y as *const _;
        assert!(eq_ref(&x, unsafe { &*y }));
    }
}
