//! Check that temporaries in the for into-iterable expr are not dropped
//! until the end of the for expr.

//@ run-pass

static mut FLAGS: u64 = 0;

struct AddFlags {
    bits: u64,
}

impl Drop for AddFlags {
    fn drop(&mut self) {
        unsafe {
            FLAGS += self.bits;
        }
    }
}

fn check_flags(expected: u64) {
    unsafe {
        let actual = FLAGS;
        FLAGS = 0;
        assert_eq!(actual, expected, "flags {}, expected {}", actual, expected);
    }
}

fn main() {
    for _ in &[AddFlags { bits: 1 }] {
        check_flags(0);
    }
    check_flags(1);
}
