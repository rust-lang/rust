//@ compile-flags: -Zunleash-the-miri-inside-of-you

// Make sure we cannot mutate globals.

static mut GLOBAL: i32 = 0;

static MUTATING_GLOBAL: () = {
    unsafe {
        GLOBAL = 99
        //~^ ERROR modifying a static's initial value
    }
};

fn main() {}
