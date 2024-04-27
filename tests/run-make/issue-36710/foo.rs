// Tests that linking to C++ code with global destructors works.

extern "C" {
    fn get() -> u32;
}

fn main() {
    let i = unsafe { get() };
    assert_eq!(i, 1234);
}
