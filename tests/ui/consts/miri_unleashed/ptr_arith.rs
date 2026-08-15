//@ compile-flags: -Zunleash-the-miri-inside-of-you

// During CTFE, we prevent pointer-to-int casts.
// Pointer comparisons are prevented in the trait system.

static PTR_INT_CAST: () = {
    let x = &0 as *const _ as usize;
    //~^ ERROR exposing pointers
    let _v = x == x;
};

static PTR_INT_TRANSMUTE: () = unsafe {
    let x: usize = std::mem::transmute(&0);
    let _v = x + 0;
    //~^ ERROR unable to turn pointer into integer
};

// I'd love to test pointer comparison, but that is not possible since
// their `PartialEq` impl is non-`const`.

fn main() {}

//~? WARN skipping const checks
