mod hey {
    pub use std::println as bitflags;
}

fn main() {
    bitflags! { //~ ERROR cannot find macro `bitflags` in this scope
        struct Flags: u32 {
            const A = 0b00000001;
        }
    }
}
//~ HELP consider importing this macro:
//       use hey::bitflags;
