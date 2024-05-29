//@ run-pass
// This test verifies that temporaries created for `while`'s and `if`
// conditions are dropped after the condition is evaluated.

struct Temporary;

static mut DROPPED: isize = 0;

impl Drop for Temporary {
    fn drop(&mut self) {
        unsafe { DROPPED += 1; }
    }
}

impl Temporary {
    fn do_stuff(&self) -> bool {true}
}

fn borrow() -> Box<Temporary> { Box::new(Temporary) }


pub fn main() {
    let mut i = 0;

    // This loop's condition
    // should call `Temporary`'s
    // `drop` 6 times.
    while borrow().do_stuff() {
        i += 1;
        unsafe { assert_eq!(DROPPED, i) }
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        if i > 5 {
            break;
        }
    }

    // This if condition should
    // call it 1 time
    if borrow().do_stuff() {
        unsafe { assert_eq!(DROPPED, i + 1) }
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    }
}
