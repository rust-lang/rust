#![no_std]
#![no_main]

extern crate alloc;

#[stem::main]
fn main(_arg: usize) -> ! {
    stem::println!("Shutting down...");
    stem::shutdown();
}
