//@ run-pass
#![feature(const_refs_to_static)]

static S: i32 = 0;
static mut S_MUT: i32 = 0;

const C1: &i32 = &S;
#[allow(unused)]
const C1_READ: () = {
    assert!(*C1 == 0);
};
const C2: *const i32 = unsafe { std::ptr::addr_of!(S_MUT) };

fn main() {
    assert_eq!(*C1, 0);
    assert_eq!(unsafe { *C2 }, 0);
    // Computing this pattern will read from an immutable static. That's fine.
    assert!(matches!(&0, C1));
}
