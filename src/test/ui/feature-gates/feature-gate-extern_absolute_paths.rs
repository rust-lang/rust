use core::default; //~ ERROR unresolved import `core`

fn main() {
    let _: u8 = ::core::default::Default(); //~ ERROR failed to resolve
}
