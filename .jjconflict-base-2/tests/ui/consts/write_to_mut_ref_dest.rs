//@run-pass

const FOO: &u32 = {
    let mut a = 42;
    {
        let b: *mut u32 = &mut a;
        unsafe { *b = 5; }
    }
    &{a}
};

fn main() {
    assert_eq!(*FOO, 5);
}
