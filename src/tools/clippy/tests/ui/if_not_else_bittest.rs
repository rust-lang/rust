//@ check-pass

#![deny(clippy::if_not_else)]

fn show_permissions(flags: u32) {
    if flags & 0x0F00 != 0 {
        println!("Has the 0x0F00 permission.");
    } else {
        println!("The 0x0F00 permission is missing.");
    }
}

fn main() {}
