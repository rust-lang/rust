#![feature(const_fn)]
#![feature(const_if_match)]
#![feature(const_unreachable_unchecked)]

const unsafe fn foo(x: u8) -> u8 {
    match x {
        42 => 34,
        _ => std::hint::unreachable_unchecked(),
    }
}

const BAR:u8 = unsafe { foo(250) };

fn main() {
    println!("{}", BAR);
}
