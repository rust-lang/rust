// compile-flags: -Zunleash-the-miri-inside-of-you

// Make sure we cannot mutate globals.

static mut GLOBAL: i32 = 0;

const MUTATING_GLOBAL: () = {
    unsafe {
        GLOBAL = 99 //~ ERROR any use of this value will cause an error
        //~^ WARN skipping const checks
        //~| WARN skipping const checks
    }
};

fn main() {}
