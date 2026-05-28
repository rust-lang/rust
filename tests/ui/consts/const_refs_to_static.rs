//@ run-pass
use std::sync::atomic::AtomicU32;

static S: i32 = 0;
static mut S_MUT: i32 = 0;

const C1: &i32 = &S;
#[allow(unused)]
const C1_READ: () = {
    assert!(*C1 == 0);
};
const C2: *const i32 = std::ptr::addr_of!(S_MUT);

static FOO: AtomicU32 = AtomicU32::new(0);
const NOT_VALID_AS_PATTERN: &'static AtomicU32 = &FOO;

fn main() {
    assert_eq!(*C1, 0);
    assert_eq!(unsafe { *C2 }, 0);
    // Computing this pattern will read from an immutable static. That's fine.
    assert!(matches!(&0, C1));
    let _val = NOT_VALID_AS_PATTERN;
}
