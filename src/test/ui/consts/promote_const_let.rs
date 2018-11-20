#![feature(const_let)]

fn main() {
    let x: &'static u32 = {
        let y = 42;
        &y //~ ERROR does not live long enough
    };
}