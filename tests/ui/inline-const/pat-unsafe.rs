// check-pass

#![warn(unused_unsafe)]
#![feature(inline_const_pat)]

const unsafe fn require_unsafe() -> usize {
    1
}

fn main() {
    unsafe {
        match () {
            const {
                require_unsafe();
                unsafe {}
                //~^ WARNING unnecessary `unsafe` block
            } => (),
        }

        match 1 {
            const {
                unsafe {}
                //~^ WARNING unnecessary `unsafe` block
                require_unsafe()
            }..=4 => (),
            _ => (),
        }
    }
}
