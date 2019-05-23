#![deny(unused_must_use)]

fn main() {
    (Ok::<(), ()>(()),); //~ ERROR unused `std::result::Result` that must be used
}
