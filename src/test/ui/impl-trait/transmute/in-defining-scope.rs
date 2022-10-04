// This causes query cycles when trying to reveal the hidden type of `foo`.
use std::mem::transmute;
fn foo() -> impl Sized {
    //~^ ERROR cycle detected when computing type
    unsafe {
        transmute::<_, u8>(foo());
        //~^ ERROR cannot transmute between types of different sizes, or dependently-sized types
    }
    0u8
}

fn main() {}
