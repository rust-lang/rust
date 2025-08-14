//@error-in-other-file: hand-coded well-known IP address
//@no-rustfix
#![warn(clippy::ip_constant)]

fn external_constant_test() {
    let _ = include!("localhost.txt");
    // lint in external file `localhost.txt`
}

fn main() {
    external_constant_test();
}
