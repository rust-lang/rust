// run-pass
// aux-build:xcrate_static_addresses.rs

// pretty-expanded FIXME #23616

extern crate xcrate_static_addresses;

use xcrate_static_addresses as other;

pub fn main() {
    other::verify_same(&other::global);
    other::verify_same2(other::global2);
}
