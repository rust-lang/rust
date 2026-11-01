//@ignore-target: windows # No socket support on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native
#![feature(gethostname)]

fn main() {
    let hostname = std::net::hostname().unwrap();
    if cfg!(miri) {
        assert_eq!(hostname, "Miri");
    }
}
