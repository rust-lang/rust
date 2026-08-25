//! Tests macro_metavars_in_unsafe with private (non-exported) macros
#![warn(clippy::macro_metavars_in_unsafe)]

macro_rules! mac {
    ($v:expr) => {
        unsafe {
            //~^ ERROR: this macro expands metavariables in an unsafe block
            dbg!($v);
        }
    };
}

fn main() {
    mac!(1);
}
